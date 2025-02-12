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
class AnalysisResult:
    reasoning: str
    retrieval_queries: List[str]

    def __str__(self) -> str:
        queries_str = "\n    ".join(self.retrieval_queries)
        return f"""Analysis Result:
Reasoning: {self.reasoning}
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

**Task:** Analyze the user's goal "{goal}" using first principles and generate effective search queries to understand it deeply.

**First Principles Analysis - Break Down the Goal:**

Think step-by-step to understand the *core* of the user's goal. Ask yourself:

1. **User's Core Need:** What is the user *really* trying to accomplish beyond their stated goal? What's their underlying need or problem?
2. **Fundamental Components:** What are the essential pieces of information needed to fully address this core need?  Think of the basic concepts and elements involved.
3. **Motivation:** *Why* is the user asking this? What are they hoping to achieve by understanding this goal? (Consider context and purpose).
4. **Verification:** How can we confirm we've truly understood and addressed the user's core need? What would demonstrate a complete answer?

**Query Generation - Information Retrieval Strategy:**

Based on your first principles analysis, create a set of search queries to gather the necessary information.  Your queries MUST:

* **Cover Key Dimensions:**  Address all essential aspects identified in your "Fundamental Components" analysis.
* **Use Varied Terminology:** Include both broad and specific terms, synonyms, and related concepts.
* **Explore Context:**  Generate queries to understand the context and related concepts of the user's goal.
* **Validate Assumptions:** Create queries to verify any assumptions you've made during your analysis.
* **Decompose Complexity:** For complex goals, break them down into sub-queries focusing on individual components.
* **Employ Search Patterns:** Use a mix of exact match keywords and semantic search phrases (natural language questions).

**Generate a5 least 3-5 High-Priority Queries:** Order these queries from most to least important for initial understanding.

**Output Json Format:**
```json
{{
    "reasoning": "Detailed first-principles analysis explaining your understanding of the user's core need, the fundamental components, and how you arrived at the retrieval queries.",
    "retrieval_queries": [
        "High-priority search query 1",
        "Next most important query",
        "...",
    ]
}}
```
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
                retrieval_queries=data["retrieval_queries"],
            )

        except (KeyError, json.JSONDecodeError, Exception) as e:
            logger.error("Parsing failed: %s, data %s", e, raw,exc_info=True)
            raise AnalysisError("Invalid analysis response") from e
