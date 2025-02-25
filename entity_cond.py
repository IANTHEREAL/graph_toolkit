import logging
import json
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session

from llm_inference.base import LLMInterface
from graph.graph_knowledge_base import GraphKnowledgeBase, SearchAction
from graph.query_analyzer import DeepUnderstandingAnalyzer
from graph.knowledge_synthesizer import KnowledgeSynthesizer

logger = logging.getLogger(__name__)


class EntityCond:
    def __init__(self, llm_client: LLMInterface, kb: GraphKnowledgeBase):
        self.llm_client = llm_client
        self.kb = kb
        self.analyzer = DeepUnderstandingAnalyzer(llm_client)
        self.synthesizer = KnowledgeSynthesizer(llm_client)

    def apply(
        self, session: Session, next_actions: List[SearchAction], **model_kwargs
    ) -> List[str]:
        knowledge_retrieved = {}
        for action in next_actions:
            logger.info(f"applying {action}")
            if action.tool == "retrieve_graph_data":
                data = self.kb.retrieve_graph_data(
                    session, action.query, action.top_k, **model_kwargs
                )
            elif action.tool == "retrieve_neighbors":
                data = self.kb.retrieve_neighbors(
                    session,
                    action.entity_ids,
                    action.query,
                    action.top_k,
                    **model_kwargs,
                )
            else:
                raise ValueError(f"Invalid tool: {action.tool}")

            for doc_id, doc in data.documents.items():
                if doc_id not in knowledge_retrieved:
                    knowledge_retrieved[doc_id] = doc

                for chunk_id, chunk in doc.chunks.items():
                    if chunk_id not in knowledge_retrieved[doc_id].chunks:
                        knowledge_retrieved[doc_id].chunks[chunk_id] = chunk
                        continue

                    existing_chunk = knowledge_retrieved[doc_id].chunks[chunk_id]
                    rel_dict = {r.id: r for r in existing_chunk.relationships}
                    for relationship in chunk.relationships:
                        rel_id = relationship.id
                        if rel_id in rel_dict:
                            rel_dict[rel_id].similarity_score = max(
                                rel_dict[rel_id].similarity_score,
                                relationship.similarity_score,
                            )
                        else:
                            rel_dict[rel_id] = relationship

                    knowledge_retrieved[doc_id].chunks[chunk_id].relationships = list(
                        rel_dict.values()
                    )

        return knowledge_retrieved

    def analyze(
        self, session: Session, query: str, top_k: int = 20, **model_kwargs
    ) -> List[str]:
        analysis_res = self.analyzer.perform(query, **model_kwargs)
        logger.info(f"Initial analysis for {query}: {analysis_res}")
        next_actions = [
            SearchAction(tool="retrieve_graph_data", query=a, top_k=top_k)
            for a in analysis_res.retrieval_queries
        ]

        docs = self.apply(session, next_actions, **model_kwargs)
        result = self.synthesizer.map_reduce_synthesis(
            query=query,
            documents=docs,
            **model_kwargs,
        )

        return result

    def store_result(
        self,
        session: Session,
        query: str,
        final_answer: str,
        related_relationships: List[Dict],
        **model_kwargs,
    ):
        return self.kb.store_synopsis_entity(
            session,
            query,
            final_answer,
            related_relationships,
            **model_kwargs,
        )
