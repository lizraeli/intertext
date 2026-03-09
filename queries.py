from typing import Any, Optional, Protocol, Sequence

from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func
from sqlalchemy import desc

from models import Novel, NovelSegment


def query_novel_segments(db: Session, novel_id: int):
    return (
        db.query(NovelSegment)
        .filter(NovelSegment.novel_id == novel_id)
        .order_by(NovelSegment.macro_block_id, NovelSegment.id)
        .all()
    )


def query_similar_by_vector(
    db: Session, embedding: list[float], theme_filter: Optional[str], limit: int
):
    query = db.query(
        Novel.title,
        Novel.author,
        NovelSegment.content,
        NovelSegment.embedding.cosine_distance(embedding).label("distance"),
    ).join(Novel, NovelSegment.novel_id == Novel.id)

    if theme_filter:
        query = query.filter(
            NovelSegment.metadata_col["primary_themes"].contains([theme_filter])
        )

    return (
        query.order_by(NovelSegment.embedding.cosine_distance(embedding))
        .limit(limit)
        .all()
    )


class RandomSegmentsRow(Protocol):
    id: int
    content: str
    metadata_col: dict[str, Any]


def query_random_segments(db: Session, count: int) -> Sequence[RandomSegmentsRow]:
    return (
        db.query(NovelSegment.id, NovelSegment.content, NovelSegment.metadata_col)
        .order_by(func.random())
        .limit(count)
        .all()
    )


class SegmentByIdRow(Protocol):
    id: int
    novel_id: int
    content: str
    metadata_col: dict[str, Any]
    title: str
    author: str
    publication_year: int | None


def query_segment_by_id(db: Session, segment_id: int) -> SegmentByIdRow | None:
    return (
        db.query(
            NovelSegment.id,
            NovelSegment.novel_id,
            NovelSegment.content,
            NovelSegment.metadata_col,
            Novel.title,
            Novel.author,
            Novel.publication_year,
        )
        .join(Novel, NovelSegment.novel_id == Novel.id)
        .filter(NovelSegment.id == segment_id)
        .first()
    )


def query_segment_embedding(db: Session, segment_id: int):
    return (
        db.query(NovelSegment.embedding).filter(NovelSegment.id == segment_id).first()
    )


class SimilarRow(Protocol):
    id: int
    content: str
    metadata_col: dict[str, Any]
    title: str
    author: str
    distance: float


def query_similar_by_segment(
    db: Session, segment_id: int, embedding: list[float], limit: int
) -> Sequence[SimilarRow]:
    return (
        db.query(
            NovelSegment.id,
            NovelSegment.content,
            NovelSegment.metadata_col,
            Novel.title,
            Novel.author,
            NovelSegment.embedding.cosine_distance(embedding).label("distance"),
        )
        .join(Novel, NovelSegment.novel_id == Novel.id)
        .filter(NovelSegment.id != segment_id)
        .order_by(NovelSegment.embedding.cosine_distance(embedding))
        .limit(limit)
        .all()
    )


def query_different_by_segment(
    db: Session, segment_id: int, embedding: list[float], limit: int
) -> Sequence[SimilarRow]:
    return (
        db.query(
            NovelSegment.id,
            NovelSegment.content,
            NovelSegment.metadata_col,
            Novel.title,
            Novel.author,
            NovelSegment.embedding.cosine_distance(embedding).label("distance"),
        )
        .join(Novel, NovelSegment.novel_id == Novel.id)
        .filter(NovelSegment.novel_id != segment_id)
        .order_by(desc(NovelSegment.embedding.cosine_distance(embedding)))
        .limit(limit)
        .all()
    )
