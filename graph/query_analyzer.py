import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import wraps

from utils.json_utils import extract_json

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
    initial_queries: List[str]

    def __str__(self) -> str:
        queries_str = "\n    ".join(self.initial_queries)
        return f"""Analysis Result:
Reasoning: {self.reasoning}
Intent:
    Action: {self.intent.action}
    Target: {self.intent.target}
    Context: {self.intent.context}
Initial Queries:
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


def get_analysis_prompt(query):
    return f"""Deeply analyze this query using first principles thinking: "{query}"

Goal: Understand the fundamental nature of the user's question by breaking it down to its most basic elements.

First Principles Analysis:
1. What is the user truly trying to achieve? (Look beyond the surface request)
2. What are the fundamental components needed to answer this?
3. Why is the user asking this question? (Consider underlying needs)
4. How would we know if we've fully answered the question?

Output Format:
{{
    "reasoning": "Deep analysis starting from first principles, explaining how we understand the user's fundamental need and how we arrived at our conclusions",
    "intent": {{
        "action": "<fundamental operation needed>",
        "target": "<core entity or concept>",
        "context": "<broader context or domain>"
    }},
    "initial_queries": [
        "concrete search phrase derived from our understanding"
    ]
}}

Key Requirements:
- Reasoning MUST start from first principles, not assumptions
- Intent MUST reflect the fundamental need, not just surface request
- All components MUST be derived from the reasoning

Remember: The goal is to understand the true nature of the question, not just its surface form.
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
                initial_queries=data["initial_queries"],
            )

        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Parsing failed: {str(e)}", exc_info=True)
            raise AnalysisError("Invalid analysis response") from e
