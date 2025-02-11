import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import wraps

from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

# ---------------------------
# Data Structures
# ---------------------------


@dataclass
class QueryIntent:
    """Derived from reasoning analysis"""

    action: str
    target: str
    context: str


@dataclass
class AnalysisResult:
    reasoning: str
    intent: QueryIntent
    retrieval_queries: List[str]

    def __str__(self) -> str:
        queries_str = "\n    ".join(self.retrieval_queries)
        return f"""Analysis Result:
Reasoning: {self.reasoning}
Intent:
    Action: {self.intent.action}
    Target: {self.intent.target}
    Context: {self.intent.context}
Retrieval Queries:
    {queries_str}"""


# ---------------------------
# Exceptions
# ---------------------------


class AnalysisError(Exception):
    """Base analysis error"""


class ValidationError(AnalysisError):
    """Causal flow validation failure"""


# ---------------------------
# Utility Functions
# ---------------------------


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    sleep_time = delay * (2 ** (attempts - 1))
                    logging.warning(f"Retry {attempts}/{max_attempts} in {sleep_time}s")
                    time.sleep(sleep_time)

        return wrapper

    return decorator


# ---------------------------
# prompt definition
# ---------------------------


def get_analysis_prompt(goal):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""Current Time: {current_time}

Deeply analyze this goal using first principles thinking: "{goal}"

Goal: Understand the fundamental nature of the user's question by breaking it down to its most basic elements.

First Principles Analysis:
1. What is the user truly trying to achieve? (Look beyond the surface request)
2. What are the fundamental components needed to answer this?
3. Why is the user asking this question? (Consider underlying needs)
4. How would we know if we've fully answered the question?

Key Requirements:
- Reasoning MUST start from first principles, not assumptions
- Intent MUST reflect the fundamental need, not just surface request
- retrieval_queries MUST include:
  * All essential search dimensions identified in the reasoning
  * Both broad and specific terminology variants
  * Potential related concepts that could provide context
  * Verification queries to validate assumptions
  * Decomposed sub-queries for complex concepts
- Generate 3-5 queries minimum, ordered by priority
- Use both exact match and semantic search patterns

Remember: The queries should form a complete information retrieval strategy based on your deep understanding.
Output Format:
{{
    "reasoning": "Deep analysis starting from first principles, explaining how we understand the user's fundamental need and how we arrived at our conclusions",
    "intent": {{
        "action": "<fundamental operation needed>",
        "target": "<core entity or concept>",
        "context": "<broader context or domain>"
    }},
    "retrieval_queries": [
        "concrete search query 1 derived from our understanding",
        "alternative phrasing for core concept",
    ]
}}
"""


# ---------------------------
# Core Analysis Class
# ---------------------------


class DeepUnderstandingAnalyzer:
    """
    Implements reasoning-first analysis flow:
    Reasoning → Intent → Search Strategy → Initial Queries
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    @retry()
    def perform(self, query, **model_kwargs):
        """Orchestrate graph construction workflow"""
        prompt = get_analysis_prompt(query)
        raw_response = self.llm_client.generate(prompt, **model_kwargs)
        return self._parse_response(raw_response)

    def _parse_response(self, raw: str):
        """Parse with causal validation"""
        try:

            json_str = extract_json(raw)
            data = json.loads(json_str)

            return AnalysisResult(
                reasoning=data["reasoning"],
                intent=QueryIntent(**data["intent"]),
                retrieval_queries=data["retrieval_queries"],
            )

        except (KeyError, json.JSONDecodeError, Exception) as e:
            logger.error("Parsing failed: %s, data %s", e, raw,exc_info=True)
            raise AnalysisError("Invalid analysis response") from e
