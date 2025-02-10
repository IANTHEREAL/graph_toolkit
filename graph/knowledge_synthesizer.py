from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import json
import logging
import datetime

from utils.json_utils import extract_json
from .graph_knowledge_base import SearchAction, DocumentData

# ---------------------------
# Data Structures
# ---------------------------


@dataclass
class Finding:
    """Represents a single knowledge finding"""

    content: str
    confidence_score: float
    is_novel: bool  # True if this is a new finding
    is_supplementary: bool  # True if this supplements existing findings
    reasoning: str  # Explanation of how this finding was derived
    source_quotes: List[Dict]

    def to_dict(self):
        return {
            "content": self.content,
            "source_quotes": self.source_quotes,
            "confidence_score": self.confidence_score,
            "is_novel": self.is_novel,
            "is_supplementary": self.is_supplementary,
            "reasoning": self.reasoning,
        }


@dataclass
class EvaluationResult:
    """Represents the evaluation of current findings"""

    is_sufficient: bool
    missing_aspects: List[str]
    next_actions: List[SearchAction]
    reasoning: str

    def to_dict(self):
        return {
            "is_sufficient": self.is_sufficient,
            "missing_aspects": self.missing_aspects,
            "next_actions": [a.to_dict() for a in self.next_actions],
            "reasoning": self.reasoning,
        }


class KnowledgeSynthesizer:
    """
    Synthesizes knowledge from search results into a list of findings
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_findings(
        self, reasoning: str, query: str, search_results: Dict, current_findings: List
    ) -> List[Finding]:
        findings = self._extract_findings(
            reasoning, query, search_results, current_findings
        )
        current_findings.extend(findings)
        return current_findings

    def _extract_findings(
        self, reasoning: str, query: str, search_results: Dict, current_findings: List
    ) -> List[Finding]:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""Current Time: {current_time}

Your goal is to analyze search results and extract findings that help answer the original query.
Focus on identifying information that directly contributes to answering the query or provides important context.

Original Query: {query}
Reasoning on Original Query: {reasoning}

Existing findings: {current_findings}

Search Results: (relationships and chunks, both of them contain doc_link that is the source links of the knowledge)
{json.dumps(search_results, indent=2)}

IMPORTANT RULES:
1. Focus on findings that are RELEVANT to answering the original query
2. EVERY finding MUST be directly supported by specific content in the Search Results
3. EVERY finding MUST include the IDs of relevant relationships and/or chunks as evidence
4. DO NOT make assumptions or extrapolations beyond what is explicitly stated in the Search Results
5. If information appears in multiple documents, use this to INCREASE confidence score
6. Confidence scores should be calculated based on:
   - Document reliability from Search Results
   - Information consistency across knowledge data (relationships/chunks)
   - Directness of evidence (direct statement vs inference)
   - Relevance to the original query
   - Number of independent sources confirming the information (+0.1 for each additional source)
7. ACTIVELY SEEK different perspectives from existing findings:
   - Look for information that presents alternative viewpoints
   - Identify nuances or exceptions to current findings
   - Consider contradictory evidence or opposing arguments
   - Highlight complementary but distinct aspects not covered in existing findings

Extract key findings that either:
1. Directly answer aspects of the original query
2. Provide essential context or background needed to understand the answer
3. Present DIFFERENT perspectives or viewpoints from existing findings
4. Challenge or qualify existing findings with new evidence
5. Add nuance or complexity to the current understanding

For each finding, provide:
- The finding content (MUST be a direct fact from Search Results that helps answer the query)
- IDs of all relevant relationships and/or chunks with doc_link that support this finding
- Reasoning explaining how this finding helps answer the query and why we can trust it
- Conservative confidence score (0.0-1.0)
- Whether it's novel or supplementary to existing findings

Output Format:
{{
    "findings": [
        {{
            "content": "precise finding with specific details",
            "source_quotes":[
                {{
                    "doc_link": "doc_link_1",
                    "chunk_ids": [
                        "chunk_id_1",
                        "chunk_id_2"
                    ],
                    "relationship_ids": [
                        "relationship_id_1",
                        "relationship_id_2"
                    ]
                }},{{
                    "doc_link": "doc_link_2",
                    "chunk_ids": [
                        "chunk_id_1",
                        "chunk_id_2"
                    ],
                    "relationship_ids": [
                        "relationship_id_1",
                        "relationship_id_2"
                    ]
                }}
            ],
            "reasoning": "explanation of how these relationships and/or chunks support this finding and why we can trust it",
            "confidence_score": 0.95,
            "is_novel": true,
            "is_supplementary": false
        }}
    ]
}}

VERIFICATION CHECKLIST:
- Each finding has supporting chunk_ids and/or relationship_ids from Search Results
- No unsupported claims or assumptions
- Confidence scores reflect evidence quality from Search Results
- New findings add diversity to existing perspectives
- Findings explore different angles or interpretations of the evidence
"""
        try:
            raw_response = self.llm.generate(prompt)
            json_str = extract_json(raw_response)
            data = json.loads(json_str)

            findings = []
            for f in data["findings"]:
                finding = Finding(
                    content=f["content"],
                    source_quotes=f["source_quotes"],
                    confidence_score=f["confidence_score"],
                    is_novel=f["is_novel"],
                    is_supplementary=f["is_supplementary"],
                    reasoning=f["reasoning"],
                )
                findings.append(finding)

            return findings

        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Failed to extract findings: {str(e)}", exc_info=True)
            raise ValueError("Invalid findings extraction response") from e

    def evaluate_findings(
        self,
        query: str,
        reasoning: str,
        current_findings: List[Finding],
        docs: Dict,
        action_history: List[SearchAction],
    ) -> EvaluationResult:
        """
        Evaluates if current findings are sufficient and suggests next search actions.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""Current Time: {current_time}
Evaluate if the current findings are sufficient to answer the original query and determine next search actions.

Original Query: {query}

Analysis Reasoning:
{reasoning}

Current Findings:
{current_findings}

Documents for current findings:
{docs}

Previous Actions:
{action_history}

Evaluation Tasks:
1. Analyze Query Completeness:
   - Compare the query's key aspects with current findings
   - Identify any gaps in knowledge
   - Consider if the findings' confidence scores are adequate
   - Check if key findings are supported by multiple independent sources
   - Compare findings with other sources' answers to validate information
   - Identify which findings need additional cross-document validation

2. Determine Sufficiency:
   - Are all key aspects of the query addressed?
   - Is the information quality sufficient?
   - Are there any logical gaps?
   - Do important findings have corroboration from multiple sources?
   - Is there agreement between our findings and other sources' answers?
   - Are there any significant claims supported by only a single source?

3. If insufficient, plan next actions choosing from:
   a) retrieve_knowledge: For exploring completely new aspects
      - Use when needing information about new concepts or topics
      - Use to find corroborating evidence from different sources
      - Use to verify important claims from additional documents
   
   b) retrieve_neighbors: For expanding from known entities
      - Use when wanting to explore connections from existing entities
      - Use to find alternative sources discussing the same entities

Output Format:
{{
    "is_sufficient": true/false,
    "missing_aspects": [
        "description of missing aspect 1",
        "description of missing aspect 2"
    ],
    "next_actions": [
        {{
            "tool": "retrieve_knowledge",
            "query": "specific search query for new information"
        }},
        {{
            "tool": "retrieve_neighbors",
            "entity_ids": [1, 2],
            "query": "focused query to guide neighbor exploration"
        }}
    ],
    "reasoning": "detailed explanation of the evaluation and action choices"
}}

Important:
- Choose tools based on the type of missing information:
  * New topics -> retrieve_knowledge
  * Entity connections -> retrieve_neighbors
  * Relationship details -> retrieve_chunks (ONLY for relationships)
- Adjust confidence scores based on agreement with other sources:
  * Increase score when multiple sources agree (+0.1 for each confirming source)
  * Decrease score when sources conflict (-0.1 for each conflicting source)
  * Consider the credibility of each source when weighing agreement
- Prioritize finding multiple independent sources for important claims
- Seek verification from different document types and sources
- When findings come from a single source, actively search for corroboration
- Consider the credibility and perspective of each source
- Avoid suggesting actions similar to previous unsuccessful attempts
- Prioritize actions that address the most critical missing aspects first
"""

        try:
            raw_response = self.llm.generate(prompt)
            json_str = extract_json(raw_response)
            eval_res = json.loads(json_str)

            next_actions = [SearchAction(**a) for a in eval_res["next_actions"]]
            return EvaluationResult(
                is_sufficient=eval_res["is_sufficient"],
                missing_aspects=eval_res["missing_aspects"],
                next_actions=next_actions,
                reasoning=eval_res["reasoning"],
            )
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Failed to evaluate findings: {str(e)}", exc_info=True)
            raise ValueError("Invalid evaluation response") from e

    def iterative_answer_synthesis(
        self, query: str, documents: Dict[str, DocumentData], reasoning: str
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
        }

        for doc_id, doc in documents.items():
            print(f"Processing document({doc_id}, {doc.doc_link})")

            prompt = f"""Current Time: {current_time}

Your task is to strategically incorporate new information while maintaining answer quality.

[STRICT REQUIREMENT]
1. FIRST verify document relevance to query: "{query}"

Original Query: {query}
Current Answer State: {answer_state["current_answer"] or "No answer yet"}
Document Source: {doc.doc_link}
Document Content: {doc.content}

Analysis Framework:
0. Relevance Check:
   - Does document content directly address {query}?

1. Value Assessment:
   - Identify novel information not in current answer
   - Detect more recent versions of existing facts
   - Find complementary perspectives/evidence
   - Verify source credibility compared to existing sources

2. Update Decision Criteria (proceed only if):
   - Contains verifiable information missing from current answer
   - Provides higher-confidence evidence for existing claims
   - Offers critical context for understanding key points
   - Presents equally valid alternative perspectives

3. Information Integration Rules:
   - Preserve superior existing information when:
     * Current evidence is from more reliable sources
     * Existing version has broader consensus
     * New data doesn't improve clarity/completeness
   - Prefer concise synthesis over redundant accumulation
   - Mark deprecated information with [Superseded] tags
   - Maintain version history in footnotes

Output Format:
{{
    "should_skip": boolean,
    "skip_reason": "Required if should_skip=true",
    "updated_answer": "Revised answer (unchanged if should_skip=true)",
    "change_breakdown": {{
        "improvements": ["List of substantive upgrades"],
        "preserved_content": ["Key maintained elements"],
        "deprecated_items": ["Outdated information replaced"]
    }},
    "confidence_impact": "+X%/-X% based on changes"
}}

Quality Assurance:
- MUST skip documents about unrelated systems/tools
- Skip if document adds <10% new relevant content
- Skip when confidence impact would be <+5%
- Require minimum two independent sources for major changes
- Preserve superior phrasing from existing answer
- Prefer older established facts over new unverified claims
- Maintain traceability through source anchors
"""
            try:
                MAX_RETRIES = 3
                for retry_count in range(MAX_RETRIES):
                    try:
                        raw_response = self.llm.generate(prompt)
                        json_str = extract_json(raw_response)
                        update_data = json.loads(json_str)

                        if update_data.get("should_skip", False):
                            print(f"Skipped document: {update_data['skip_reason']}")
                            continue

                        print(
                            f"Quality improvements: {update_data['change_breakdown']['improvements']}"
                        )

                        # Update answer state only if not skipped
                        answer_state["current_answer"] = update_data["updated_answer"]

                        # Track evolution with more granular metadata
                        answer_state["history"].append(
                            {
                                "doc_link": doc.doc_link,
                                "changes": update_data["change_breakdown"],
                                "confidence_change": update_data["confidence_impact"],
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
                                f"Failed to process document {doc.doc_link} after {MAX_RETRIES} retries: {str(e)}",
                                exc_info=True,
                            )
                            continue  # Move to next document
                        logging.warning(
                            f"Retry {retry_count + 1}/{MAX_RETRIES} - Failed to process document {doc.doc_link}: {str(e)}"
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
        }
