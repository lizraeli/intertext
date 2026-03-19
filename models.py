from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    Table,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Any

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
    characters = relationship(
        "NovelCharacter", back_populates="novel", cascade="all, delete-orphan"
    )
    places = relationship(
        "NovelPlace", back_populates="novel", cascade="all, delete-orphan"
    )
    moods = relationship(
        "NovelMood", back_populates="novel", cascade="all, delete-orphan"
    )


segment_characters = Table(
    "segment_characters",
    Base.metadata,
    Column(
        "segment_id",
        Integer,
        ForeignKey("novel_segments.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "character_id",
        Integer,
        ForeignKey("novel_characters.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class NovelCharacter(Base):
    __tablename__ = "novel_characters"

    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)

    novel: Mapped[Novel] = relationship("Novel", back_populates="characters")
    segments: Mapped[list["NovelSegment"]] = relationship(
        "NovelSegment",
        secondary=segment_characters,
        back_populates="characters",
    )

    __table_args__ = (
        UniqueConstraint("novel_id", "name", name="uq_novel_character_name"),
    )


class NovelPlace(Base):
    __tablename__ = "novel_places"
    __table_args__ = (UniqueConstraint("novel_id", "name", name="uq_novel_place_name"),)

    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)

    novel: Mapped[Novel] = relationship("Novel", back_populates="places")
    segments: Mapped[list["NovelSegment"]] = relationship(
        "NovelSegment", back_populates="place"
    )


class NovelMood(Base):
    __tablename__ = "novel_moods"
    __table_args__ = (UniqueConstraint("novel_id", "name", name="uq_novel_mood_name"),)

    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)

    novel: Mapped[Novel] = relationship("Novel", back_populates="moods")
    segments: Mapped[list["NovelSegment"]] = relationship(
        "NovelSegment", back_populates="mood"
    )


class NovelSegment(Base):
    __tablename__ = "novel_segments"

    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False, index=True)
    place_id: Mapped[int] = Column(
        Integer,
        ForeignKey("novel_places.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    mood_id: Mapped[int] = Column(
        Integer,
        ForeignKey("novel_moods.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    macro_block_id = Column(Integer, nullable=False)
    start_index = Column(Integer, nullable=False)
    end_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    metadata_col: Mapped[dict[str, Any]] = Column("metadata", JSONB, nullable=False)
    embedding: Mapped[list[float] | None] = Column(Vector(3072))

    novel: Mapped[Novel] = relationship("Novel", back_populates="segments")
    characters: Mapped[list["NovelCharacter"]] = relationship(
        "NovelCharacter",
        secondary=segment_characters,
        back_populates="segments",
    )
    place: Mapped["NovelPlace"] = relationship("NovelPlace", back_populates="segments")
    mood: Mapped["NovelMood"] = relationship("NovelMood", back_populates="segments")
