from uuid import UUID, uuid4
from sqlalchemy import Column, String, Text, JSON, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from setting.db import Base
from tidb_vector.sqlalchemy import VectorType


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(PostgresUUID, primary_key=True, default=uuid4)
    hash = Column(String(64), nullable=False)
    text = Column(Text, nullable=False)
    meta = Column(JSON, default={})
    embedding = Column(VectorType(1536), comment="hnsw(distance=cosine)")
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    source_uri = Column(String(512), nullable=True)

    def __init__(self, embedding_dims: int = 1536, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Column(
            VectorType(embedding_dims), comment="hnsw(distance=cosine)"
        )

    def __repr__(self):
        return f"<Chunk(id={self.id}, hash={self.hash})>"
