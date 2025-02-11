from sqlalchemy import Column, Text, JSON, DateTime, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tidb_vector.sqlalchemy import VectorType
from setting.db import Base


def get_relationship_model(
    table_name: str = "relationships", vector_length: int = 1536
):
    """
    Returns a custom Entity model with the specified table name and vector length.
    """

    entity_table_name = "entities_" + table_name.split("_")[1]

    class Relationship(Base):
        __tablename__ = table_name

        id = Column(Integer, primary_key=True, autoincrement=True)
        description = Column(Text, nullable=False)
        meta = Column(JSON, default={})
        source_entity_id = Column(
            "source_entity_id", Integer, ForeignKey(f"{entity_table_name}.id")
        )
        target_entity_id = Column(
            "target_entity_id", Integer, ForeignKey(f"{entity_table_name}.id")
        )
        weight = Column(Integer, default=0)
        last_modified_at = Column(DateTime)
        document_id = Column(Integer, nullable=True)
        chunk_id = Column(String(36), nullable=True)
        description_vec = Column(
            VectorType(vector_length), comment="hnsw(distance=cosine)"
        )

        # Relationships
        source_entity = relationship(
            "Entity", foreign_keys=[source_entity_id], lazy="joined"
        )
        target_entity = relationship(
            "Entity", foreign_keys=[target_entity_id], lazy="joined"
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

    return Relationship
