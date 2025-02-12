from typing import Dict, List
import json
import logging
import datetime

from utils.json_utils import extract_json
from .graph_knowledge_base import DocumentData

logger = logging.getLogger(__name__)


class KnowledgeSynthesizer:
    """
    Synthesizes knowledge from search results into a list of findings
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def iterative_answer(
        self,
        query: str,
        documents: Dict[str, DocumentData],
        **model_kwargs,
    ) -> Dict:
        """
        Iteratively process documents to build and refine an answer to the query.

        Args:
            query: The original query to answer
            documents: Dictionary of documents sorted by relevance
            reasoning: Initial reasoning about the query

        Returns:
            Dict containing the final answer and its evolution history
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        answer_state = {
            "current_answer": "",
            "history": [],  # Track how the answer evolved
            "relationships": [],
        }

        for doc_id, doc in documents.items():
            logger.info(f"Processing document({doc_id}, {doc.doc_link})")
            relationships_list = []
            for chunk in doc.chunks.values():
                for rel in chunk.relationships:
                    relationships_list.append(
                        f"relationship_id: {rel.id}, description: {rel.source_entity['name']} -> {rel.relationship} -> {rel.target_entity['name']}"
                    )

            prompt = f"""Current Time: {current_time}

Your goal is to update and improve an answer to the query using information from the provided document.

**Instructions:**

1. **Information Retention:** 
   - NEVER remove information from the existing answer unless the new document EXPLICITLY contradicts it
   - When documents conflict, prefer information from more authoritative/recent sources
2. **Answer Improvement:** 
   - Add new information ONLY when the document provides:  
     a) More precise values (e.g., "10s" → "15s")  
     b) Additional constraints (e.g., "requires TLS 1.2+")  
     c) Clear corrections (e.g., "previous docs had a typo")
     d) Qualifying conditions (e.g., "only when cluster_version > 6.0")
3. **Source Grounding:**  
   - All new information MUST come directly from the "Document Content"
   - Cite using `[Source:{doc.doc_link}]` for key claims
4. **Contradiction Handling:**
   - Only overwrite existing info if the document EXPLICITLY states it's incorrect
   - When overwriting, clearly note the correction (e.g., "Correction: The timeout is actually 15s [Source:...]")
5. **Skip Irrelevant Docs:** 
   - Skip if the document adds no new information to any aspect of the current answer

**Original Query:** {query}
**Current Answer State (English):** {answer_state["current_answer"] or "No answer yet"}

**Document Analysis:**
- **Source:** {doc.doc_link}
- **Document Content:** {doc.content}

**Output Json Format (ENGLISH ONLY):**
```json
{{
    "should_skip": boolean,
    "skip_reason": "Optional reason if skipping",
    "updated_answer": "Revised English answer with [Source:doc_link] citations"
    "commit_message": "commit message for the answer"
}}
```
"""

            try:
                MAX_RETRIES = 3
                for retry_count in range(MAX_RETRIES):
                    try:
                        print(prompt)
                        raw_response = self.llm.generate(prompt, **model_kwargs)
                        json_str = extract_json(raw_response)
                        update_data = json.loads(json_str)

                        logger.info(
                            "Commit Detail: %s", json.dumps(update_data, indent=2)
                        )

                        if update_data.get("should_skip", False) is True:
                            logger.warning(
                                f"Skipped document: {update_data['skip_reason']}"
                            )
                            break

                        # Update answer state only if not skipped
                        answer_state["current_answer"] = update_data["updated_answer"]
                        for chunk in doc.chunks.values():
                            for rel in chunk.relationships:
                                answer_state["relationships"].append(rel.to_dict())

                        # Track evolution with more granular metadata
                        answer_state["history"].append(
                            {
                                "doc_link": doc.doc_link,
                                "commit_message": update_data["commit_message"],
                                "update_decision": (
                                    "Applied"
                                    if not update_data["should_skip"]
                                    else "Skipped"
                                ),
                                "decision_reason": update_data.get("skip_reason", ""),
                            }
                        )
                        break

                    except (KeyError, json.JSONDecodeError) as e:
                        if retry_count == MAX_RETRIES - 1:  # Last retry
                            logging.error(
                                f"Failed to process document {doc.doc_link} after {MAX_RETRIES} retries: {str(e)}. response: {raw_response}",
                                exc_info=True,
                            )
                            continue  # Move to next document
                        logging.warning(
                            f"Retry {retry_count + 1}/{MAX_RETRIES} - Failed to process document {doc.doc_link}: {str(e)}. response: {raw_response}"
                        )
            except Exception as e:
                logging.error(
                    f"Failed to process document {doc.doc_link}: {str(e)}",
                    exc_info=True,
                )
                continue  # Move to next document

        return {
            "final_answer": answer_state["current_answer"],
            "evolution_history": answer_state["history"],
            "relationships": answer_state["relationships"],
        }

    def answer_on_document(self, query: str, doc: DocumentData, **model_kwargs) -> Dict:
        """Process individual documents"""
        try:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt = f"""Current Time: {current_time}
Generate an answer segment using ONLY this document. Return should_skip=true if irrelevant.

Rules:
1. Use ONLY facts explicitly stated in the document
2. Include ANY information that helps answer the query, even if partial
3. Include [Source:{doc.doc_link}] citations in the answer segment
4. Relevance criteria:
   - RELEVANT: Document contains ANY facts that help answer the query
   - NOT RELEVANT: Document has NO facts that help answer the query
   - When in doubt about relevance, please don't include the information

Query: {query}
Document Content: {doc.content}  # Prevent context overflow

Output JSON:
```json
{{
    "should_skip": bool,  # true ONLY if NO relevant information found
    "doc_answer": "Answer text with citations based on the document content",
    "key_facts": [
        "Specific fact text",
        ...
    ]
}}
```
"""

            raw_response = self.llm.generate(prompt, **model_kwargs)
            json_str = extract_json(raw_response)
            response = json.loads(json_str)

            logger.info(
                "answer on document %s result: %s",
                doc.doc_link,
                json.dumps(response, indent=2),
            )

            if not response.get("should_skip"):
                return {
                    "doc_link": doc.doc_link,
                    "answer_segment": response["doc_answer"],
                    "key_facts": response["key_facts"],
                }

        except Exception as e:
            logger.error(
                "Map phase failed for %s:%s, response: %s",
                doc.doc_link,
                str(e),
                raw_response,
            )

        return None

    def synthesize_answer(
        self, query: str, map_results: List[Dict], **model_kwargs
    ) -> Dict:
        """Enhanced synthesis answer from map results"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Group segments and facts by document
        doc_segments = [
            {
                "doc_link": res["doc_link"],
                "answer_segment": res["answer_segment"],
                "key_facts": res.get("key_facts", []),
            }
            for res in map_results
            if res
        ]

        prompt = f"""Current Time: {current_time}

Your task is to synthesize a comprehensive answer from multiple document segments.

**Synthesis Rules:**
1. **Deduplication:** 
  - Merge duplicate information
  - Keep the most complete version when similar facts appear
2. **Conflict Resolution:**
  - When conflicts occur, prefer information that:
    a) Appears in multiple sources
    b) Comes from more authoritative sources
    c) Has more recent timestamps or From newer versions
3. **Source Attribution:**
  - Maintain all source citations [Source:link]
  - When multiple sources support the same fact, combine citations. 
    - Example: Fact description [Source: doc1][Source: doc2]
4. **Structure:**
   - Organize information logically
   - Use clear sections and bullet points
   - Ensure smooth transitions between topics

**Original Query:** {query}

**Documents to Synthesize:**
{json.dumps(doc_segments, indent=2)}

Output JSON Format:
```json
{{
    "final_answer": "Structured answer with source citations",
    "sources_adapted": ["doc_link1", "doc_link2", ...],
}}
```
"""

        try:
            MAX_RETRIES = 3
            for retry_count in range(MAX_RETRIES):
                try:
                    raw_response = self.llm.generate(prompt, **model_kwargs)
                    json_str = extract_json(raw_response)
                    result = json.loads(json_str)

                    return {
                        "final_answer": result["final_answer"],
                        "sources_adapted": list(
                            set(result["sources_adapted"])
                        ),  # Deduplicate sources
                    }

                except (json.JSONDecodeError, KeyError, Exception) as e:
                    if retry_count == MAX_RETRIES - 1:  # Last retry
                        raise
                    logger.warning(
                        f"Retry {retry_count + 1}/{MAX_RETRIES} - Synthesis failed: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}", exc_info=True)
            return {
                "final_answer": "Failed to synthesize final answer. Please refer to original segments.",
                "sources_adapted": [],
                "error": str(e),
            }

    def map_reduce_synthesis(
        self, query: str, documents: Dict[str, DocumentData], **model_kwargs
    ) -> Dict:
        """Full map-reduce synthesis workflow"""

        # Phase 1: Sequential Mapping
        map_results = []
        relationships = []
        for doc in documents.values():
            res = self.answer_on_document(query, doc, **model_kwargs)
            if res:
                for chunk in doc.chunks.values():
                    for rel in chunk.relationships:
                        relationships.append(rel.to_dict())
                logger.info("answer on document %s result: %s", doc.doc_link, res)
                map_results.append(res)
            else:
                logger.warning("Skipped irrelevant document %s", doc.doc_link)

        # Phase 2: Result Reduction
        if len(map_results) > 0:
            reduce_result = self.synthesize_answer(query, map_results, **model_kwargs)
        else:
            reduce_result = {
                "final_answer": None,
                "sources_adapted": [],
                "relationships": [],
            }

        return {
            **reduce_result,
            "relationships": relationships,
            "map_results": map_results,
        }
