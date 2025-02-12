import logging
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple
from sqlmodel import Session
from sqlalchemy import text

from models.entity import get_entity_model
from models.relationship import get_relationship_model

from graph.chunk_filter import ChunkFilter
from llm_inference.embedding import (
    get_text_embedding,
    get_entity_description_embedding,
    get_entity_metadata_embedding,
)
from json_utils import extract_json

logger = logging.getLogger(__name__)


@dataclass
class SearchAction:
    """Represents a next search action"""

    tool: Literal["retrieve_graph_data", "retrieve_neighbors", "retrieve_documents"]
    query: Optional[str] = None  # For retrieve_knowledge and retrieve_neighbors
    entity_ids: Optional[List[int]] = None  # For retrieve_neighbors
    doc_link: Optional[str] = None  # For retrieve_knowledge

    def to_dict(self):
        return {
            "tool": self.tool,
            "query": self.query,
            "entity_ids": self.entity_ids,
            "doc_link": self.doc_link,
        }

    def __str__(self):
        if self.tool == "retrieve_graph_data":
            return f"SearchAction(tool={self.tool}, query={self.query})"
        elif self.tool == "retrieve_neighbors":
            return f"SearchAction(tool={self.tool}, query={self.query}, entity_ids={self.entity_ids})"
        else:
            raise ValueError(f"Invalid tool: {self.tool}")


@dataclass
class RelationshipData:
    id: int
    relationship: str
    chunk_id: int
    document_id: int
    doc_link: str
    source_entity: Dict[str, Any]
    target_entity: Dict[str, Any]
    similarity_score: float

    def to_dict(self):
        return {
            "id": self.id,
            "relationship": self.relationship,
            "chunk_id": self.chunk_id,
            "doc_link": self.doc_link,
            "document_id": self.document_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "similarity_score": self.similarity_score,
        }


@dataclass
class ChunkData:
    id: str = ""
    content: Optional[str] = None
    relationships: List[RelationshipData] = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "relationships": [
                relationship.to_dict() for relationship in self.relationships
            ],
        }


@dataclass
class DocumentData:
    id: int
    chunks: Dict[int, ChunkData]  # key: chunk_id
    content: Optional[str] = None  # document content
    doc_link: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "doc_link": self.doc_link,
            "chunks": {
                chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()
            },
        }


@dataclass
class GraphRetrievalResult:
    documents: Dict[str, DocumentData]  # key: document id

    def to_dict(self):
        return {"documents": {id: doc.to_dict() for id, doc in self.documents.items()}}


class GraphKnowledgeBase:
    def __init__(
        self,
        llm_client,
        entity_table_name,
        relationship_table_name,
        chunk_table_name,
        document_table_name="documents",
    ):
        self._entity_table = entity_table_name
        self._relationship_table = relationship_table_name
        self._chunk_table = chunk_table_name
        self._document_table = document_table_name
        self.chunk_filter = ChunkFilter(llm_client, 5, 1)
        self._entity_model = get_entity_model(self._entity_table)
        self._relationship_model = get_relationship_model(self._relationship_table)

    def retrieve_graph_data(
        self,
        session: Session,
        query_text: str,
        top_k: int = 20,
        similarity_threshold: float = 0.5,
        **model_kwargs,
    ) -> GraphRetrievalResult:
        query_embedding = get_text_embedding(query_text)

        # Query similar relationships using raw SQL
        relationship_sql = text(
            f"""
            SELECT r.id, r.description, r.chunk_id, r.document_id,
                se.id as source_id, se.name as source_name, se.description as source_description,
                te.id as target_id, te.name as target_name, te.description as target_description,
                JSON_UNQUOTE(JSON_EXTRACT(r.meta, '$.source_uri')) AS doc_link,
                (1 - VEC_COSINE_DISTANCE(r.description_vec, :query_embedding)) as similarity
            FROM {self._relationship_table} r
            JOIN {self._entity_table} se ON r.source_entity_id = se.id
            JOIN {self._entity_table} te ON r.target_entity_id = te.id
            ORDER BY similarity DESC
            LIMIT :limit
            """
        )

        # Execute queries
        relationships = []
        start_time = time.time()
        relationship_results = session.execute(
            relationship_sql,
            {
                "query_embedding": str(query_embedding),
                "threshold": similarity_threshold,
                "limit": top_k,
            },
        ).fetchall()
        logger.info(f"query relationships use {time.time() - start_time} seconds")

        # Process relationship results
        for row in relationship_results:
            if row.similarity < similarity_threshold:
                continue
            relationships.append(
                {
                    "id": row.id,
                    "relationship": row.description,
                    "chunk_id": row.chunk_id,
                    "doc_link": row.doc_link,
                    "document_id": row.document_id,
                    "source_entity": {
                        "id": row.source_id,
                        "name": row.source_name,
                        "description": row.source_description,
                    },
                    "target_entity": {
                        "id": row.target_id,
                        "name": row.target_name,
                        "description": row.target_description,
                    },
                    "similarity_score": row.similarity,
                }
            )

        chunks = self._get_chunks(
            session, [row.get("chunk_id") for row in relationships]
        )

        filtered_results = self.chunk_filter.filter_chunks(
            query_text, chunks, **model_kwargs
        )

        # Get document IDs from relevant chunks
        relevant_chunk_ids = {
            result.chunk_id
            for result in filtered_results
            if result.relevant and result.confidence > 0.6
        }

        result = GraphRetrievalResult(documents={})
        documents_id = set()
        for chunk in chunks:
            chunk_id = chunk["id"]
            if chunk_id not in relevant_chunk_ids:
                continue
            for relationship in relationships:
                if relationship["chunk_id"] == chunk_id:
                    doc_id = relationship["document_id"]
                    if doc_id not in result.documents:
                        if doc_id in documents_id:
                            continue
                        documents_id.add(doc_id)
                        result.documents[doc_id] = DocumentData(
                            id=doc_id, chunks={}, doc_link=chunk["doc_link"]
                        )
                    if chunk_id not in result.documents[doc_id].chunks:
                        result.documents[doc_id].chunks[chunk_id] = ChunkData(
                            id=chunk_id, content=chunk["content"], relationships=[]
                        )
                    result.documents[doc_id].chunks[chunk_id].relationships.append(
                        RelationshipData(**relationship)
                    )

        if len(documents_id) > 0:
            documents = self._get_documents(session, list(documents_id))
            for doc_id, doc in documents.items():
                if doc_id in result.documents:
                    result.documents[doc_id].content = doc["content"]

        return result

    def retrieve_neighbors(
        self,
        session: Session,
        entities_ids: List[int],
        query: str,
        max_depth: int = 1,
        max_neighbors: int = 20,
        similarity_threshold: float = 0.5,
    ) -> GraphRetrievalResult:
        query_embedding = get_text_embedding(query)

        # Track visited nodes and discovered paths
        all_visited = set(entities_ids)
        current_level_nodes = set(entities_ids)
        neighbors = []

        for depth in range(max_depth):
            if not current_level_nodes:
                break

            # Query relationships using raw SQL
            relationship_sql = text(
                f"""
                SELECT r.id, r.description, r.chunk_id, r.document_id, r.source_entity_id, r.target_entity_id,
                       se.name as source_name, se.description as source_description,
                       te.name as target_name, te.description as target_description,
                       JSON_UNQUOTE(JSON_EXTRACT(r.meta, '$.source_uri')) AS doc_link,
                       (1 - VEC_COSINE_DISTANCE(r.description_vec, :query_embedding)) as similarity
                FROM {self._relationship_table} r
                JOIN {self._entity_table} se ON r.source_entity_id = se.id
                JOIN {self._entity_table} te ON r.target_entity_id = te.id
                WHERE r.source_entity_id IN :current_nodes
                   OR r.target_entity_id IN :current_nodes
                ORDER BY similarity DESC
                LIMIT :limit
                """
            )

            relationships = session.execute(
                relationship_sql,
                {
                    "query_embedding": str(query_embedding),
                    "current_nodes": current_level_nodes,
                    "threshold": similarity_threshold,
                    "limit": max_neighbors,
                },
            ).fetchall()

            next_level_nodes = set()

            for row in relationships:
                if row.similarity < similarity_threshold:
                    continue

                # Determine direction and connected entity
                if row.source_entity_id in current_level_nodes:
                    connected_id = row.target_entity_id
                else:
                    connected_id = row.source_entity_id

                # Skip if already visited
                if connected_id in all_visited:
                    continue

                neighbors.append(
                    {
                        "id": row.id,
                        "relationship": row.description,
                        "doc_link": row.doc_link,
                        "chunk_id": row.chunk_id,
                        "document_id": row.document_id,
                        "source_entity": {
                            "id": row.source_entity_id,
                            "name": row.source_name,
                            "description": row.source_description,
                        },
                        "target_entity": {
                            "id": row.target_entity_id,
                            "name": row.target_name,
                            "description": row.target_description,
                        },
                        "similarity_score": row.similarity,
                    }
                )

                next_level_nodes.add(connected_id)
                all_visited.add(connected_id)

            current_level_nodes = next_level_nodes

        # Sort and limit results
        neighbors.sort(key=lambda x: x["similarity_score"], reverse=True)
        relationships = neighbors[:max_neighbors]
        chunks = self.get_chunks(
            session, [row.get("chunk_id") for row in relationships]
        )

        filtered_results = self.chunk_filter.filter_chunks(query, chunks)
        # Get document IDs from relevant chunks
        relevant_chunk_ids = {
            result.chunk_id
            for result in filtered_results
            if result.relevant and result.confidence > 0.6
        }

        # Convert to GraphRetrievalResult
        result = GraphRetrievalResult(documents={})
        documents_id = set()
        for chunk in chunks:
            chunk_id = chunk["id"]
            if chunk_id not in relevant_chunk_ids:
                continue
            for relationship in relationships:
                if relationship["chunk_id"] == chunk_id:
                    doc_id = relationship["document_id"]
                    if doc_id not in result.documents:
                        if doc_id in documents_id:
                            continue
                        documents_id.add(doc_id)
                        result.documents[doc_id] = DocumentData(
                            id=doc_id, chunks={}, doc_link=chunk["doc_link"]
                        )
                    if chunk_id not in result.documents[doc_id].chunks:
                        result.documents[doc_id].chunks[chunk_id] = ChunkData(
                            id=chunk_id, content=chunk["content"], relationships=[]
                        )
                    result.documents[doc_id].chunks[chunk_id].relationships.append(
                        RelationshipData(**relationship)
                    )

        if len(documents_id) > 0:
            documents = self._get_documents(session, list(documents_id))
            for doc_id, doc in documents.items():
                if doc_id in result.documents:
                    result.documents[doc_id].content = doc["content"]

        return result

    def _get_chunks(
        self,
        session: Session,
        chunk_ids: List[int],
    ) -> List[Dict[str, Any]]:
        chunks_sql = text(
            f"""SELECT c.id, c.text, c.document_id, c.source_uri AS doc_link FROM {self._chunk_table} c WHERE c.id IN :ids"""
        )

        chunks = session.execute(chunks_sql, {"ids": chunk_ids}).fetchall()

        return [
            {
                "id": chunk.id,
                "content": chunk.text,
                "document_id": chunk.document_id,
                "doc_link": chunk.doc_link,
            }
            for chunk in chunks
        ]

    def _get_documents(
        self, session: Session, doc_ids: List[int]
    ) -> List[Dict[str, Any]]:
        doc_sql = text(
            f"""SELECT id, content, source_uri as doc_link FROM {self._document_table} WHERE id IN :ids"""
        )
        docs = session.execute(doc_sql, {"ids": doc_ids}).fetchall()
        return {
            doc.id: {
                "id": doc.id,
                "content": doc.content,
                "doc_link": doc.doc_link,
            }
            for doc in docs
        }

    def get_document(self, session: Session, doc_link: str) -> List[Dict[str, Any]]:
        doc_sql = text(
            f"""
            SELECT id, content, source_uri as doc_link
            FROM {self._document_table}
            WHERE source_uri = :doc_link
        """
        )

        doc = session.execute(doc_sql, {"doc_link": doc_link}).fetchone()
        return GraphRetrievalResult(
            documents={
                doc.id: DocumentData(
                    id=doc.id, content=doc.content, doc_link=doc.doc_link, chunks={}
                )
            }
        )

    def store_synopsis_entity(
        self,
        db_session: Session,
        original_query: str,
        final_answer: str,
        related_relationships: List[Dict],
        **model_kwargs,
    ) -> Tuple[int, List[int]]:
        # Optimize query using LLM
        optimize_prompt = f"""Rewrite the following technical query to be clearer, more concise, and easier to retrieve.
The rewritten query should:
1. Maintain technical accuracy
2. Use standard technical terminology
3. Be structured as a "How to..." or "Why..." question

Original query: {original_query}

Return only the rewritten query without any explanation in a json format.
Json format:
{{
    "optimized_query": "rewritten query"
}}
"""

        try:
            response = self.llm_client.generate(prompt=optimize_prompt, **model_kwargs)
            response_str = extract_json(response)
            response_json = json.loads(response_str)
            optimized_query = response_json["optimized_query"]
        except Exception as e:
            logger.error(f"Failed to optimize query: {str(e)}")
            optimized_query = original_query

        # Generate embeddings for entity
        try:
            # Combine name and description for description_vec
            description_vec = get_entity_description_embedding(
                optimized_query, final_answer
            )

            # Generate meta_vec from metadata
            meta_data = {
                "original_query": original_query,
                "referenced_relationships": [
                    rel["id"] for rel in related_relationships
                ],
            }
            meta_vec = get_entity_metadata_embedding(meta_data)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            description_vec = None
            meta_vec = None

        # Create synopsis entity
        synopsis_entity = self._entity_model(
            name=optimized_query,
            description=final_answer,
            entity_type="synopsis",
            meta={
                "original_query": original_query,
                "referenced_relationships": [
                    rel["id"] for rel in related_relationships
                ],
            },
            description_vec=description_vec,
            meta_vec=meta_vec,
        )

        try:
            db_session.add(synopsis_entity)
            db_session.flush()  # Get the generated ID
        except Exception as e:
            logger.error(f"Failed to save synopsis entity: {str(e)}")
            db_session.rollback()

        logger.info(f"Stored synopsis entity: {synopsis_entity.id}")

        # Create new relationships linking synopsis to source entities
        new_relationships = []
        for rel in related_relationships:
            rel_description = f"Summary connection for: {optimized_query}..."
            try:
                # Generate embedding for relationship description
                rel_vec = get_text_embedding(rel_description)
            except Exception as e:
                logger.error(f"Failed to generate relationship embedding: {str(e)}")
                rel_vec = None

            new_rel = self._relationship_model(
                description=rel_description,
                source_entity_id=synopsis_entity.id,
                target_entity_id=rel["source_entity"]["id"],
                document_id=rel["document_id"],
                chunk_id=rel["chunk_id"],
                meta={
                    "referenced_relationship_id": rel["id"],
                    "source_uri": rel["doc_link"],
                },
                description_vec=rel_vec,
            )
            new_relationships.append(new_rel)

        try:
            db_session.bulk_save_objects(new_relationships)
            db_session.flush()  # Add this line to flush before commit
            relationship_ids = [
                rel.id for rel in new_relationships
            ]  # Get IDs after flush
            db_session.commit()
        except Exception as e:
            logger.error(f"Failed to save relationships: {str(e)}")
            db_session.rollback()
            relationship_ids = []  # Return empty list if failed

        return {
            "synopsis_entity_id": synopsis_entity.id,
            "new_relationship_ids": relationship_ids,  # Use the collected IDs
        }
