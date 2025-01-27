import openai
import time

from typing import Any, Dict, List, Optional
from sqlmodel import Session, select
from sqlalchemy.orm import defer, joinedload
from sqlalchemy import or_, desc, text

from setting.db import SessionLocal

embedding_model = openai.OpenAI()


def get_text_embedding(text: str, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return (
        embedding_model.embeddings.create(input=[text], model=model).data[0].embedding
    )


class GraphKnowledgeBase:
    def __init__(self, entity_table_name, relationship_table_name, chunk_table_name):
        self._entity_table = entity_table_name
        self._relationship_table = relationship_table_name
        self._chunk_table = chunk_table_name

    def retrieve_graph_data(
        self,
        session: Session,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.6,
    ) -> Dict[str, List[Dict[str, Any]]]:
        query_embedding = get_text_embedding(query_text)

        # Query similar relationships using raw SQL
        relationship_sql = text(
            f"""
            SELECT r.id, r.description,
                se.id as source_id, se.name as source_name, se.description as source_description,
                te.id as target_id, te.name as target_name, te.description as target_description,
                JSON_EXTRACT(r.meta, '$.source_uri') AS doc_link,
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

        chunks = self.get_chunks_by_relationships(
            session, [row.get("id") for row in relationships]
        )

        return {"chunks": chunks, "relationships": relationships}

    def retrieve_neighbors(
        self,
        session: Session,
        entities_ids: List[int],
        query: str,
        max_depth: int = 1,
        max_neighbors: int = 20,
        similarity_threshold: float = 0.6,
    ) -> Dict[str, List[Dict]]:
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
                SELECT r.id, r.description, r.source_entity_id, r.target_entity_id,
                       se.name as source_name, se.description as source_description,
                       te.name as target_name, te.description as target_description,
                       JSON_EXTRACT(r.meta, '$.source_uri') AS doc_link,
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
        chunks = self.get_chunks_by_relationships(
            session, [row.get("id") for row in relationships]
        )
        return {"relationships": relationships, "chunks": chunks}

    def get_chunks_by_relationships(
        self,
        session: Session,
        relationships_ids: List[int],
    ) -> List[Dict[str, Any]]:
        chunks_sql = text(
            f"""
    WITH chunk_ids AS (
        SELECT DISTINCT chunk_id
        FROM {self._relationship_table}
        WHERE id IN :ids AND chunk_id IS NOT NULL
    )
    SELECT 
        c.id,
        c.text,
        c.document_id,
        c.source_uri AS doc_link
    FROM {self._chunk_table} c
    WHERE c.id IN (SELECT chunk_id FROM chunk_ids)"""
        )

        chunks = session.execute(chunks_sql, {"ids": relationships_ids}).fetchall()

        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "doc_link": chunk.doc_link,
            }
            for chunk in chunks
        ]
