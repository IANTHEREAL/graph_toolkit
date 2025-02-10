import json
from typing import List, Dict
import logging
from dataclasses import dataclass

from json_utils import extract_json_array

logger = logging.getLogger(__name__)


@dataclass
class FilteredChunk:
    chunk_id: str
    is_relevant: bool
    confidence: float
    reasoning: str


class ChunkFilter:
    def __init__(self, llm_client, batch_size: int = 5, max_workers: int = 3):
        self.llm = llm_client
        self.batch_size = batch_size
        self.max_workers = max_workers

    def filter_chunks(self, query: str, chunks: List[Dict]) -> List[FilteredChunk]:
        """Filter chunks using LLM to determine relevance to query"""
        filtered_chunks = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            filtered_chunks.extend(self._process_batch(query, batch))

        return filtered_chunks

    def _process_batch(self, query: str, chunks: List[Dict]) -> List[FilteredChunk]:
        """Process a batch of chunks"""
        prompt = self._build_filter_prompt(query, chunks)

        try:
            logger.info("Filter chunks %s", [chunk["id"] for chunk in chunks])
            response = self.llm.generate(prompt=prompt)
            filtered_chunks = self._parse_filter_response(
                response, [c["id"] for c in chunks]
            )
            return filtered_chunks
        except Exception as e:
            logger.error(f"Error filtering chunks: {str(e)}")
            # Return all chunks as relevant in case of error
            return [
                FilteredChunk(
                    chunk_id=chunk["id"],
                    is_relevant=True,
                    confidence=0.5,
                    reasoning="Error in filtering, keeping chunk as relevant",
                )
                for chunk in chunks
            ]

    def _build_filter_prompt(self, query: str, chunks: List[Dict]) -> str:
        chunks_text = "\n\n".join(
            [f"Chunk(chunk_id={chunk['id']}):\n{chunk['content']}" for chunk in chunks]
        )

        return f"""Analyze if the following text chunks are relevant to answering this query. Consider:
- Direct relevance to query subject matter
- Presence of key entities/relationships mentioned in query
- Specificity of information relative to query
- Timeliness of information if query is time-sensitive

Query: {query}

{chunks_text}

For each chunk, provide STRICT evaluation with clear justification.
Return a JSON Array format, each item in the array is a JSON object for a chunk with the following fields:
```json
[
    {{
        "chunk_id": "1",  // MUST match chunk_id from input
        "is_relevant": true/false,  // REQUIRED boolean
        "confidence": 0.75,  // REQUIRED float between 0-1
        "reasoning": "Specific quotes/terms matching query"  // REQUIRED concrete evidence
    }},
    ...
] 
```

Important: 
- If uncertain, mark is_relevant=false with low confidence
- Never invent information not in the chunk
- Validate chunk_id matches exactly"""

    def _parse_filter_response(
        self, response: str, chunk_ids: List[str]
    ) -> List[FilteredChunk]:
        """Parse LLM response into FilteredChunk objects"""
        try:
            results = json.loads(extract_json_array(response))
        except Exception as e:
            logger.error(
                "Error parsing filter response %s, %s", response, e, exc_info=True
            )
            return [
                FilteredChunk(
                    chunk_id=chunk_id,
                    is_relevant=False,  # Changed from True to fail safe
                    confidence=0.0,
                    reasoning=f"Error parsing response: {str(e)}",
                )
                for chunk_id in chunk_ids
            ]

        # Create lookup for validation
        valid_ids = set(chunk_ids)

        if isinstance(results, dict):
            results = [results]

        filtered_chunks = []
        for result in results:
            logger.info("Filter Eval Result %s", result)
            try:
                chunk_id = result.get("chunk_id")  # Force string type
                if chunk_id in valid_ids:
                    filtered_chunks.append(
                        FilteredChunk(
                            chunk_id=chunk_id,
                            is_relevant=bool(result["is_relevant"]),
                            confidence=float(result["confidence"]),
                            reasoning=result["reasoning"].strip(),
                        )
                    )
                else:
                    logger.warning("Received invalid chunk_id: %s", chunk_id)
            except Exception as e:
                logger.error("Invalid result format %s, %s", result, e, exc_info=True)
                # Add fallback entry
                filtered_chunks.append(
                    FilteredChunk(
                        chunk_id=chunk_id,
                        is_relevant=False,
                        confidence=0.0,
                        reasoning=f"Invalid response format: {str(e)}",
                    )
                )

        return filtered_chunks
