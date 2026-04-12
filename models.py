from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    ForeignKey,
    Table,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Any, Optional

from database import Base


class Novel(Base):
    __tablename__ = "novels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    publication_year: Mapped[int | None] = mapped_column(Integer)

    segments: Mapped[list["NovelSegment"]] = relationship(
        "NovelSegment", back_populates="novel", cascade="all, delete-orphan"
    )
    characters: Mapped[list["NovelCharacter"]] = relationship(
        "NovelCharacter", back_populates="novel", cascade="all, delete-orphan"
    )
    places: Mapped[list["NovelPlace"]] = relationship(
        "NovelPlace", back_populates="novel", cascade="all, delete-orphan"
    )
    moods: Mapped[list["NovelMood"]] = relationship(
        "NovelMood", back_populates="novel", cascade="all, delete-orphan"
    )
    themes: Mapped[list["NovelTheme"]] = relationship(
        "NovelTheme", back_populates="novel", cascade="all, delete-orphan"
    )
    chapters: Mapped[list["NovelChapter"]] = relationship(
        "NovelChapter", back_populates="novel", cascade="all, delete-orphan"
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    novel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("novels.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    novel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("novels.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    novel: Mapped[Novel] = relationship("Novel", back_populates="places")
    segments: Mapped[list["NovelSegment"]] = relationship(
        "NovelSegment", back_populates="place"
    )


class NovelMood(Base):
    __tablename__ = "novel_moods"
    __table_args__ = (UniqueConstraint("novel_id", "name", name="uq_novel_mood_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    novel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("novels.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    novel: Mapped[Novel] = relationship("Novel", back_populates="moods")
    segments: Mapped[list["NovelSegment"]] = relationship(
        "NovelSegment", back_populates="mood"
    )


class NovelChapter(Base):
    __tablename__ = "novel_chapters"
    __table_args__ = (
        UniqueConstraint("novel_id", "block_index", name="uq_novel_chapter_block"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    novel_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("novels.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    block_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)

    novel: Mapped[Novel] = relationship("Novel", back_populates="chapters")
    segments: Mapped[list["NovelSegment"]] = relationship(
        "NovelSegment", back_populates="chapter"
    )


class NovelTheme(Base):
    __tablename__ = "novel_themes"
    __table_args__ = (UniqueConstraint("novel_id", "name", name="uq_novel_theme_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    novel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("novels.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    novel: Mapped[Novel] = relationship("Novel", back_populates="themes")
    segments: Mapped[list["SegmentTheme"]] = relationship(
        "SegmentTheme", back_populates="theme"
    )


class SegmentTheme(Base):
    """Association between a segment and a novel theme"""

    __tablename__ = "segment_themes"

    segment_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("novel_segments.id", ondelete="CASCADE"),
        primary_key=True,
    )
    theme_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("novel_themes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    intensity: Mapped[float] = mapped_column(Float, nullable=False)
    tone: Mapped[float] = mapped_column(Float, nullable=False)
    manifestation: Mapped[str] = mapped_column(Text, nullable=False)

    segment: Mapped["NovelSegment"] = relationship(
        "NovelSegment", back_populates="themes"
    )
    theme: Mapped["NovelTheme"] = relationship("NovelTheme", back_populates="segments")


class SegmentAudio(Base):
    """Audio alignment data for a segment"""

    __tablename__ = "segment_audio"

    segment_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("novel_segments.id", ondelete="CASCADE"),
        primary_key=True,
    )
    audio_key: Mapped[str] = mapped_column(String(512), nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    words: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

    segment: Mapped["NovelSegment"] = relationship(
        "NovelSegment", back_populates="audio"
    )


class NovelSegment(Base):
    __tablename__ = "novel_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    novel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("novels.id"), nullable=False, index=True
    )
    place_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("novel_places.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    mood_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("novel_moods.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    chapter_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("novel_chapters.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    start_index: Mapped[int] = mapped_column(Integer, nullable=False)
    end_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_col: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(3072))

    novel: Mapped[Novel] = relationship("Novel", back_populates="segments")
    characters: Mapped[list["NovelCharacter"]] = relationship(
        "NovelCharacter",
        secondary=segment_characters,
        back_populates="segments",
    )
    place: Mapped["NovelPlace"] = relationship("NovelPlace", back_populates="segments")
    mood: Mapped["NovelMood"] = relationship("NovelMood", back_populates="segments")
    chapter: Mapped["NovelChapter"] = relationship(
        "NovelChapter", back_populates="segments"
    )
    themes: Mapped[list["SegmentTheme"]] = relationship(
        "SegmentTheme",
        back_populates="segment",
        cascade="all, delete-orphan",
    )
    audio: Mapped[Optional["SegmentAudio"]] = relationship(
        "SegmentAudio",
        back_populates="segment",
        uselist=False,
        cascade="all, delete-orphan",
    )
