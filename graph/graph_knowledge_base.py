import openai
import time
from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Literal
from sqlmodel import Session
from sqlalchemy import text

from graph.chunk_filter import ChunkFilter

embedding_model = openai.OpenAI()


def get_text_embedding(text: str, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return (
        embedding_model.embeddings.create(input=[text], model=model).data[0].embedding
    )


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
        elif self.tool == "retrieve_documents":
            return f"SearchAction(tool={self.tool}, query={self.query})"
        else:
            raise ValueError(f"Invalid tool: {self.tool}")


@dataclass
class RelationshipData:
    id: int
    relationship: str
    chunk_id: int
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

    def retrieve_graph_data(
        self,
        session: Session,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.6,
    ) -> GraphRetrievalResult:
        query_embedding = get_text_embedding(query_text)

        # Query similar relationships using raw SQL
        relationship_sql = text(
            f"""
            SELECT r.id, r.description, r.chunk_id,
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
        print("query relationships use", time.time() - start_time)

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

        result = GraphRetrievalResult(documents={})
        for chunk in chunks:
            chunk_id = chunk["id"]
            for relationship in relationships:
                if relationship["chunk_id"] == chunk_id:
                    doc_link = relationship["doc_link"]
                    if doc_link not in result.documents:
                        result.documents[chunk["document_id"]] = DocumentData(
                            id=chunk["document_id"], chunks={}, doc_link=doc_link
                        )
                    if chunk_id not in result.documents[doc_link].chunks:
                        result.documents[doc_link].chunks[chunk_id] = ChunkData(
                            id=chunk_id, content=chunk["content"], relationships=[]
                        )
                    result.documents[doc_link].chunks[chunk_id].relationships.append(
                        RelationshipData(**relationship)
                    )

        return result

    def retrieve_neighbors(
        self,
        session: Session,
        entities_ids: List[int],
        query: str,
        max_depth: int = 1,
        max_neighbors: int = 20,
        similarity_threshold: float = 0.6,
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
                SELECT r.id, r.description, r.chunk_id, r.source_entity_id, r.target_entity_id,
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
                    "limit": max_neighbors * 2,
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

        # Convert to GraphRetrievalResult
        result = GraphRetrievalResult(documents={})
        for chunk in chunks:
            chunk_id = chunk["id"]
            for relationship in relationships:
                if relationship["chunk_id"] == chunk_id:
                    doc_link = relationship["doc_link"]
                    if doc_link not in result.documents:
                        result.documents[chunk["document_id"]] = DocumentData(
                            id=chunk["document_id"], doc_link=doc_link, chunks={}
                        )
                    if chunk_id not in result.documents[doc_link].chunks:
                        result.documents[doc_link].chunks[chunk_id] = ChunkData(
                            id=chunk_id, content=chunk["content"], relationships=[]
                        )
                    result.documents[doc_link].chunks[chunk_id].relationships.append(
                        RelationshipData(**relationship)
                    )

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

    def retrieve_documents(
        self,
        session: Session,
        query_text: str,
        top_k: int = 50,
        similarity_threshold: float = 0.5,
    ) -> GraphRetrievalResult:
        query_embedding = get_text_embedding(query_text)

        # First get relationships with similarity scores
        relationship_sql = text(
            f"""
            SELECT
                r.id, r.description, r.chunk_id,
                se.id as source_id, se.name as source_name, se.description as source_description,
                te.id as target_id, te.name as target_name, te.description as target_description,
                r.document_id,
                (1 - VEC_COSINE_DISTANCE(r.description_vec, :query_embedding)) as similarity
            FROM {self._relationship_table} r
            JOIN {self._entity_table} se ON r.source_entity_id = se.id
            JOIN {self._entity_table} te ON r.target_entity_id = te.id
            WHERE (1 - VEC_COSINE_DISTANCE(r.description_vec, :query_embedding)) >= :threshold
            ORDER BY similarity DESC
            LIMIT :limit
            """
        )

        relationship_results = session.execute(
            relationship_sql,
            {
                "query_embedding": str(query_embedding),
                "threshold": similarity_threshold,
                "limit": top_k,
            },
        ).fetchall()

        # Get chunks for filtering
        chunk_ids = [row.chunk_id for row in relationship_results]
        if not chunk_ids:
            return GraphRetrievalResult(documents={})

        chunks = self._get_chunks(session, chunk_ids)

        # Filter chunks using LLM
        filtered_results = self.chunk_filter.filter_chunks(query_text, chunks)

        # Get document IDs from relevant chunks
        relevant_chunk_ids = {
            result.chunk_id
            for result in filtered_results
            if result.is_relevant and result.confidence > 0.6
        }

        # Get document IDs from filtered chunks
        doc_ids = []
        for chunk in chunks:
            if chunk["id"] in relevant_chunk_ids:
                doc_ids.append(chunk["document_id"])

        if not doc_ids:
            return GraphRetrievalResult(documents={})

        # Query documents
        docs_sql = text(
            f"""
            SELECT id, content, source_uri as doc_link
            FROM {self._document_table}
            WHERE id IN :doc_ids
            """
        )

        result = GraphRetrievalResult(documents={})
        for doc in session.execute(docs_sql, {"doc_ids": doc_ids}):
            result.documents[doc.id] = DocumentData(
                id=doc.id, content=doc.content, doc_link=doc.doc_link, chunks={}
            )

        return result

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
