import enum
from typing import List, Dict
from sqlalchemy import Column, Integer, String, Text, JSON, Enum, Index
from setting.db import Base
from tidb_vector.sqlalchemy import VectorType


class EntityType(str, enum.Enum):
    original = "original"
    synopsis = "synopsis"

    def __str__(self):
        return self.value


class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(512), nullable=False)
    description = Column(Text, nullable=False)
    meta = Column(JSON, default={})
    entity_type = Column(Enum(EntityType), default=EntityType.original)
    synopsis_info = Column(JSON, nullable=True)
    description_vec = Column(VectorType(1536), comment="hnsw(distance=cosine)")
    meta_vec = Column(VectorType(1536), comment="hnsw(distance=cosine)")

    __table_args__ = (Index("idx_entity_type", "entity_type"),)

    def __init__(self, embedding_dims: int = 1536, **kwargs):
        super().__init__(**kwargs)
        self.description_vec = Column(
            VectorType(embedding_dims), comment="hnsw(distance=cosine)"
        )
        self.meta_vec = Column(
            VectorType(embedding_dims), comment="hnsw(distance=cosine)"
        )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False

    def screenshot(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "meta": self.meta,
            "entity_type": self.entity_type,
            "synopsis_info": self.synopsis_info,
        }

    def __repr__(self):
        return f"<Entity(id={self.id}, name={self.name}, type={self.entity_type})>"
