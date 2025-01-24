import json

from typing import List, Dict, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session
import logging

from json_utils import extract_json
from models.entity import get_entity_model
from llm_inference.base import LLMInterface

logger = logging.getLogger(__name__)

class EntityUp:
    def __init__(
        self, 
        db_session: Session, 
        llm_client: LLMInterface,
        entity_table_name: str = "entities_150001",
        chunk_table_name: str = "chunks_150001",
        relationship_table_name: str = "relationships_150001"
    ):
        self.db_session = db_session
        self.llm_client = llm_client
        self.entity_table = entity_table_name
        self.chunk_table = chunk_table_name
        self.relationship_table = relationship_table_name
        self._entity_model = get_entity_model(entity_table_name)

    def get_derived_entities(self, batch_size: int = 100, offset: int = 0) -> List[Dict]:
        """
        Two-step fetch:
        1. Get batch of entities
        2. Get all related chunks for these entities
        """
        # First get the batch of entities
        entity_query = text(f"""
            SELECT DISTINCT
                e.id as entity_id,
                e.name as entity_name,
                e.description as entity_desc,
                e.meta as entity_meta
            FROM {self.entity_table} e
            WHERE e.description LIKE 'Derived from from relationship:%'
            ORDER BY e.id
            LIMIT :limit OFFSET :offset
        """)
        
        entities = [dict(row) for row in self.db_session.execute(entity_query, {
            "limit": batch_size, 
            "offset": offset
        }).mappings()]
        
        if not entities:
            return []
        
        # Then get all related chunks for these entities
        entity_ids = [e['entity_id'] for e in entities]
        if len(entity_ids) == 1:
            # SQLAlchemy IN clause needs at least two values
            entity_ids.append(entity_ids[0])
        
        chunks_query = text(f"""
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
        """)
        
        chunks_data = [dict(row) for row in self.db_session.execute(
            chunks_query,
            {"entity_ids": tuple(entity_ids)}
        ).mappings()]
        
        # Organize chunks by entity
        entity_chunks = {}
        for chunk in chunks_data:
            # Add chunk to source entity's list
            source_id = chunk['source_entity_id']
            if source_id in entity_ids:
                if source_id not in entity_chunks:
                    entity_chunks[source_id] = set()
                entity_chunks[source_id].add(tuple(chunk.items()))
            
            # Add chunk to target entity's list
            target_id = chunk['target_entity_id']
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
            entity_id = entity['entity_id']
            if entity_id in entity_chunks:
                entity_data = entity.copy()
                entity_data['chunks'] = entity_chunks[entity_id]
                result.append(entity_data)
            else:
                # Entity has no chunks - might want to log this case
                logger.warning(f"Entity {entity_id} has no associated chunks")
                result.append(entity)
        
        return result

    def enhance_entity(self, entity_data: Dict) -> Optional[Dict]:
        """
        Use LLM to enhance entity information based on all related chunks.
        """
        # Prepare context from all chunks
        chunks_context = []
        for idx, chunk in enumerate(entity_data.get('chunks', []), 1):
            chunks_context.append(f"""
Context {idx}:
- Relationship Description: {chunk['relationship_desc']}
- Relationship Metadata: {chunk['relationship_meta']}
- Source Text: {chunk['chunk_text']}
""")
        
        chunks_context_str = "\n".join(chunks_context)
        
        prompt = f"""You are a knowledge expert assistant. Your task is to enhance the information about an entity based on its context.

Current Entity Information:
- Name: {entity_data['entity_name']}
- Current Description: {entity_data['entity_desc']}
- Current Metadata: {entity_data['entity_meta']}

Available Context:
{chunks_context_str}

Please analyze this information and generate an enhanced entity description. Focus on:
1. Technical accuracy and completeness
2. Clear and concise description
3. Relevant metadata extraction

Return a JSON object with the enhanced entity information:

```json
{{
"name": "...",
"description": "...",
"meta": {{}}
}}
```
"""
        try:
            response = self.llm_client.generate(prompt=prompt)
            json_str = extract_json(response)
            json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\r\t')
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

            entity.name = enhanced_data['name']
            entity.description = enhanced_data['description']
            entity.meta = enhanced_data['meta']
            
            self.db_session.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating entity {entity_id}: {str(e)}")
            self.db_session.rollback()
            return False

    def process_batch(self, batch_size: int = 100) -> Tuple[int, int]:
        """
        Process a batch of entities.
        Returns tuple of (success_count, error_count)
        """
        success_count = 0
        error_count = 0
        offset = 0

        while True:
            entities = self.get_derived_entities(batch_size, offset)
            if not entities:
                break

            for entity_data in entities:
                try:
                    print(f"process entity {entity_data['entity_name']}({entity_data['entity_id']})")
                    enhanced_data = self.enhance_entity(entity_data)
                    if enhanced_data and self.update_entity(entity_data['entity_id'], enhanced_data):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error processing entity {entity_data['entity_id']}: {str(e)}")
                    error_count += 1

            offset += batch_size
            break

        return success_count, error_count
