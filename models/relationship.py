from datetime import datetime
from typing import Optional, Any, List, Dict
from uuid import UUID

from sqlalchemy import Column, Text, JSON, DateTime, Integer, ForeignKey, String, Index
from sqlalchemy.orm import relationship
from tidb_vector.sqlalchemy import VectorType
from setting.db import Base


class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(Text, nullable=False)
    meta = Column(JSON, default={})
    source_entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    target_entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    weight = Column(Integer, default=0)
    last_modified_at = Column(DateTime)
    document_id = Column(Integer, nullable=True)
    chunk_id = Column(String(36), nullable=True)
    description_vec = Column(VectorType(1536), comment="hnsw(distance=cosine)")

    # Relationships
    source_entity = relationship(
        "Entity", foreign_keys=[source_entity_id], lazy="joined"
    )
    target_entity = relationship(
        "Entity", foreign_keys=[target_entity_id], lazy="joined"
    )

    def __init__(self, embedding_dims: int = 1536, **kwargs):
        super().__init__(**kwargs)
        self.description_vec = Column(
            VectorType(embedding_dims), comment="hnsw(distance=cosine)"
        )

    def __hash__(self):
        return hash(self.id)

    def screenshot(self):
        return {
            "id": self.id,
            "description": self.description,
            "meta": self.meta,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "weight": self.weight,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
        }

    def __repr__(self):
        return f"<Relationship(id={self.id}, source={self.source_entity_id}, target={self.target_entity_id})>"
