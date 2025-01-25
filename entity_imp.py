import json
from typing import List, Dict, Optional, Tuple, Callable, Any
from sqlalchemy import text
from sqlalchemy.orm import Session
import logging

from json_utils import extract_json
from models.entity import get_entity_model
from llm_inference.base import LLMInterface
from entity_agg import count_tokens

logger = logging.getLogger(__name__)


class EntityUp:
    def __init__(
        self,
        db_session: Session,
        llm_client: LLMInterface,
        entity_table_name: str = "entities_150001",
        chunk_table_name: str = "chunks_150001",
        relationship_table_name: str = "relationships_150001",
    ):
        self.db_session = db_session
        self.llm_client = llm_client
        self.entity_table = entity_table_name
        self.chunk_table = chunk_table_name
        self.relationship_table = relationship_table_name
        self._entity_model = get_entity_model(entity_table_name)

    def get_derived_entities(
        self, batch_size: int = 100, offset: int = 0
    ) -> List[Dict]:
        """
        Two-step fetch:
        1. Get batch of entities
        2. Get all related chunks for these entities
        """
        # First get the batch of entities
        entity_query = text(
            f"""
            SELECT DISTINCT
                e.id as entity_id,
                e.name as entity_name,
                e.description as entity_desc,
                e.meta as entity_meta
            FROM {self.entity_table} e
            WHERE e.description LIKE 'Derived from from relationship:%'
            ORDER BY e.id
            LIMIT :limit OFFSET :offset
        """
        )

        entities = [
            dict(row)
            for row in self.db_session.execute(
                entity_query, {"limit": batch_size, "offset": offset}
            ).mappings()
        ]

        if not entities:
            return []

        # Then get all related chunks for these entities
        entity_ids = [e["entity_id"] for e in entities]
        if len(entity_ids) == 1:
            # SQLAlchemy IN clause needs at least two values
            entity_ids.append(entity_ids[0])

        chunks_query = text(
            f"""
            SELECT 
                r.source_entity_id,
                r.target_entity_id,
                r.id as relationship_id,
                r.description as relationship_desc,
                r.meta as relationship_meta,
                r.chunk_id,
                c.text as chunk_text
            FROM {self.relationship_table} r
            JOIN {self.chunk_table} c ON r.chunk_id = c.id
            WHERE r.source_entity_id IN :entity_ids 
               OR r.target_entity_id IN :entity_ids
        """
        )

        chunks_data = [
            dict(row)
            for row in self.db_session.execute(
                chunks_query, {"entity_ids": tuple(entity_ids)}
            ).mappings()
        ]

        # Organize chunks by entity
        entity_chunks = {}
        for chunk in chunks_data:
            # Add chunk to source entity's list
            source_id = chunk["source_entity_id"]
            if source_id in entity_ids:
                if source_id not in entity_chunks:
                    entity_chunks[source_id] = set()
                entity_chunks[source_id].add(tuple(chunk.items()))

            # Add chunk to target entity's list
            target_id = chunk["target_entity_id"]
            if target_id in entity_ids:
                if target_id not in entity_chunks:
                    entity_chunks[target_id] = set()
                entity_chunks[target_id].add(tuple(chunk.items()))

        # Convert sets back to lists of dicts for further processing
        entity_chunks = {
            entity_id: [dict(items) for items in chunks]
            for entity_id, chunks in entity_chunks.items()
        }

        # Combine entity data with their chunks
        result = []
        for entity in entities:
            entity_id = entity["entity_id"]
            if entity_id in entity_chunks:
                entity_data = entity.copy()
                entity_data["chunks"] = entity_chunks[entity_id]
                result.append(entity_data)
            else:
                # Entity has no chunks - might want to log this case
                logger.warning(f"Entity {entity_id} has no associated chunks")
                result.append(entity)

        return result

    def get_poor_quality_entities(
        self, batch_size: int = 100, offset: int = 0, **model_kwargs
    ) -> List[Dict]:
        """
        Scan through entities and return those with poor information quality.
        First evaluates basic entity information, then fetches related data only for poor quality entities.

        Args:
            batch_size: Number of entities to process in one batch
            offset: Number of entities to skip
            quality_threshold: Minimum quality score threshold (0-100)

        Returns:
            List[Dict]: List of poor quality entities with their complete data
        """
        # First get basic entity information
        entity_query = text(
            f"""
            SELECT DISTINCT
                e.id as entity_id,
                e.name as entity_name,
                e.description as entity_desc,
                e.meta as entity_meta
            FROM {self.entity_table} e
            ORDER BY e.id
            LIMIT :limit OFFSET :offset
        """
        )

        entities = [
            dict(row)
            for row in self.db_session.execute(
                entity_query, {"limit": batch_size, "offset": offset}
            ).mappings()
        ]

        if not entities:
            return []

        # Get chunks for all entities
        entity_ids = [e["entity_id"] for e in entities]
        if len(entity_ids) == 1:
            # SQLAlchemy IN clause needs at least two values
            entity_ids.append(entity_ids[0])

        chunks_query = text(
            f"""
            SELECT 
                r.source_entity_id,
                r.target_entity_id,
                r.id as relationship_id,
                r.description as relationship_desc,
                r.meta as relationship_meta,
                r.chunk_id,
                c.text as chunk_text
            FROM {self.relationship_table} r
            JOIN {self.chunk_table} c ON r.chunk_id = c.id
            WHERE r.source_entity_id IN :entity_ids 
               OR r.target_entity_id IN :entity_ids
        """
        )

        chunks_data = [
            dict(row)
            for row in self.db_session.execute(
                chunks_query, {"entity_ids": tuple(entity_ids)}
            ).mappings()
        ]

        # Organize chunks by entity
        entity_chunks = {}
        for chunk in chunks_data:
            # Add chunk to source entity's list
            source_id = chunk["source_entity_id"]
            if source_id in entity_ids:
                if source_id not in entity_chunks:
                    entity_chunks[source_id] = set()
                entity_chunks[source_id].add(tuple(chunk.items()))

            # Add chunk to target entity's list
            target_id = chunk["target_entity_id"]
            if target_id in entity_ids:
                if target_id not in entity_chunks:
                    entity_chunks[target_id] = set()
                entity_chunks[target_id].add(tuple(chunk.items()))

        # Evaluate quality with complete data
        quality_threshold = model_kwargs.pop("quality_threshold", 70.0)
        poor_quality_entities = []
        for entity in entities:
            entity_id = entity["entity_id"]
            # Add chunks to entity data
            if entity_id in entity_chunks:
                entity["chunks"] = [dict(items) for items in entity_chunks[entity_id]]
            else:
                entity["chunks"] = []
                logger.warning(f"Entity {entity_id} has no associated chunks")

            try:
                print(entity)
                quality_assessment = self.evaluate_entity_quality(entity)
                print(quality_assessment)

                if not quality_assessment:
                    logger.warning(
                        f"Could not evaluate quality for entity {entity_id}, including by default"
                    )
                    poor_quality_entities.append(entity)
                    continue

                if quality_assessment["quality_score"] < quality_threshold:
                    entity["quality_assessment"] = quality_assessment
                    poor_quality_entities.append(entity)
                    logger.info(
                        f"Entity {entity_id} needs improvement "
                        f"(score: {quality_assessment['quality_score']})"
                    )

            except Exception as e:
                logger.error(f"Error evaluating entity {entity_id}: {str(e)}")
                poor_quality_entities.append(entity)

        return poor_quality_entities

    def evaluate_entity_quality(self, entity_data: Dict) -> Dict:
        """
        Use LLM to evaluate the quality of entity information and determine if enhancement is needed.
        Returns evaluation results including quality analysis and improvement suggestions.
        """
        # Format chunks information
        chunks_context = []
        for chunk in entity_data.get("chunks", []):
            chunks_context.append(
                f"""- Relationship: {chunk['relationship_desc']}
  Metadata: {chunk['relationship_meta']}
  Text: {chunk['chunk_text']}"""
            )
        chunks_text = "\n".join(chunks_context)

        prompt = f"""You are an expert entity information quality assessor. Analyze the following entity information and evaluate its quality.

Entity Information:
- Name: {entity_data['entity_name']}
- Description: {entity_data['entity_desc']}
- Metadata: {entity_data['entity_meta']}

Related Context:
{chunks_text}

Please analyze this entity information carefully. Consider:
1. Is the description comprehensive and clear?
2. Does the metadata contain all necessary structured information?
3. Is the information properly integrated with its context?
4. Are there any gaps or inconsistencies?

First provide your detailed analysis, then score the entity quality.

Return your assessment in this JSON format:
{{
    "analysis": {{
        "strengths": ["strength1", "strength2", ...],
        "weaknesses": ["weakness1", "weakness2", ...],
        "missing_information": ["missing1", "missing2", ...],
    }},
    "quality_score": <0-100>,
    "needs_enhancement": <boolean>,
    "reasoning": "detailed explanation of your scoring and enhancement recommendation",
    "improvement_suggestions": ["suggestion1", "suggestion2", ...]
}}
"""
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                options={
                    "num_ctx": 8092,
                    "num_gpu": 80,
                    "temperature": 0.1,
                },
            )
            return json.loads(extract_json(response))
        except Exception as e:
            logger.error(
                f"Error evaluating entity {entity_data['entity_id']}: {str(e)}"
            )
            return None

    def enhance_entity(
        self, entity_data: Dict, only_count_token: bool = False, **model_kwargs
    ) -> Optional[Dict]:
        """
        Use LLM to enhance entity information based on all related chunks.
        """
        # Prepare context from all chunks
        chunks_context = []
        for idx, chunk in enumerate(entity_data.get("chunks", []), 1):
            chunks_context.append(
                f"""
Context {idx}:
- Relationship Description: {chunk['relationship_desc']}
- Relationship Metadata: {chunk['relationship_meta']}
- Source Text: {chunk['chunk_text']}
"""
            )

        chunks_context_str = "\n".join(chunks_context)

        prompt = f"""You are a knowledge expert assistant. Your task is to enhance the information about an entity based on its context.

Current Entity Information:
- Name: {entity_data['entity_name']}
- Current Description: {entity_data['entity_desc']}
- Current Metadata: {entity_data['entity_meta']}

Available Context:
{chunks_context_str}

Please analyze this information and generate an enhanced entity description. Your goal is to:
1. Description should:
   - Provide a comprehensive narrative that captures all key information needed to understand the entity
   - Include essential context, relationships, and characteristics
   - Be clear, concise, and well-structured
   - Enable users to understand the entity without needing to refer to metadata

2. Metadata should:
   - Contain detailed attributes and specific data points
   - Include all technical specifications and measurements
   - Store structured information for detailed analysis
   - Capture relationships and connections with other entities
   - Preserve any numerical or categorical data

3. Ensure all information is technically accurate and complete
4. No information should be lost - all relevant details must be captured in either description or metadata


Return a JSON object with the enhanced entity information:

```json
{{
"name": "...",
"description": "...",
"meta": {{}}
}}
```
"""
        if only_count_token:
            model = model_kwargs.get("model", "gpt-4o")
            return count_tokens(prompt, model)

        try:
            response = self.llm_client.generate(prompt=prompt, **model_kwargs)
            json_str = extract_json(response)
            json_str = "".join(
                char for char in json_str if ord(char) >= 32 or char in "\r\t"
            )
            result = json.loads(json_str)
            return result
        except Exception as e:
            logger.error(f"Error enhancing entity {entity_data['entity_id']}: {str(e)}")
            return None

    def update_entity(self, entity_id: int, enhanced_data: Dict) -> bool:
        """
        Update entity with enhanced information.
        """
        try:
            entity = self.db_session.query(self._entity_model).get(entity_id)
            if not entity:
                logger.error(f"Entity {entity_id} not found")
                return False

            entity.name = enhanced_data["name"]
            entity.description = enhanced_data["description"]
            entity.meta = enhanced_data["meta"]

            self.db_session.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating entity {entity_id}: {str(e)}")
            self.db_session.rollback()
            return False

    def process_batch(
        self,
        get_entities_func: Callable[[int, int, Any], List[Dict]],
        batch_size: int = 100,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Process a batch of entities using provided entity fetching function.

        Args:
            get_entities_func: Function to fetch entities, should accept (batch_size, offset, **kwargs)
                         and return List[Dict] of entities
            batch_size: Number of entities to process in batch
            **kwargs: Additional arguments to pass to get_entities_func

        Returns:
            Tuple[int, int]: (success_count, error_count)
        """
        success_count = 0
        error_count = 0
        offset = 0

        while True:

            # Get entities using provided function with additional kwargs
            entities = get_entities_func(batch_size, offset)
            if not entities:
                break

            print(entities)
            return

            for entity_data in entities:
                try:
                    print(
                        f"Processing entity {entity_data['entity_name']}({entity_data['entity_id']})"
                    )

                    if "quality_assessment" in entity_data:
                        logger.info(
                            f"Quality assessment for entity {entity_data['entity_id']}: "
                            f"Score={entity_data['quality_assessment']['quality_score']}, "
                            f"Reasoning={entity_data['quality_assessment']['reasoning']}"
                        )

                    token_count = self.enhance_entity(
                        entity_data, only_count_token=True
                    )
                    if token_count > 16384:
                        logger.warning(
                            f"Token count {token_count} exceeds limit for entity {entity_data['entity_id']}"
                        )
                        continue

                    model_args = self._get_model_args(token_count)
                    enhanced_data = self.enhance_entity(entity_data, **model_args)

                    if enhanced_data and self.update_entity(
                        entity_data["entity_id"], enhanced_data
                    ):
                        success_count += 1
                        logger.info(
                            f"Successfully enhanced entity {entity_data['entity_id']}"
                        )
                    else:
                        error_count += 1
                        logger.error(
                            f"Failed to enhance entity {entity_data['entity_id']}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing entity {entity_data['entity_id']}: {str(e)}"
                    )
                    error_count += 1

            offset += batch_size

        logger.info(
            f"Batch processing completed. Successes: {success_count}, Errors: {error_count}"
        )
        return success_count, error_count

    def _get_model_args(self, token_count: int) -> Dict:
        """Helper method to determine model arguments based on token count"""
        if token_count > 7000:
            return {
                "options": {
                    "num_ctx": token_count + 1500,
                    "num_gpu": 80,
                    "num_predict": 10000,
                    "temperature": 0.1,
                }
            }
        return {
            "options": {
                "num_ctx": 8092,
                "num_gpu": 80,
                "num_predict": 10000,
                "temperature": 0.1,
            }
        }
