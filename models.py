from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from database import Base


class Novel(Base):
    __tablename__ = "novels"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=False)
    publication_year = Column(Integer)

    segments = relationship(
        "NovelSegment", back_populates="novel", cascade="all, delete-orphan"
    )


class NovelSegment(Base):
    __tablename__ = "novel_segments"

    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False, index=True)

    macro_block_id = Column(Integer, nullable=False)
    start_index = Column(Integer, nullable=False)
    end_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    metadata_col = Column("metadata", JSONB, nullable=False)
    embedding = Column(Vector(3072))

    novel = relationship("Novel", back_populates="segments")
