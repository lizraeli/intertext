from typing import Any, Optional, Protocol, Sequence

from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql.expression import func
from sqlalchemy import desc

from models import (
    Novel,
    NovelCharacter,
    NovelMood,
    NovelPlace,
    NovelSegment,
    NovelTheme,
    SegmentTheme,
)
from llm_schemas import ThemeAnnotation


def query_novel_segments(db: Session, novel_id: int):
    return (
        db.query(NovelSegment)
        .options(
            selectinload(NovelSegment.characters),
            selectinload(NovelSegment.place),
            selectinload(NovelSegment.mood),
            selectinload(NovelSegment.themes).selectinload(SegmentTheme.theme),
        )
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


def query_segment_by_id(db: Session, segment_id: int) -> NovelSegment | None:
    return (
        db.query(NovelSegment)
        .options(
            selectinload(NovelSegment.characters),
            selectinload(NovelSegment.place),
            selectinload(NovelSegment.mood),
            selectinload(NovelSegment.themes).selectinload(SegmentTheme.theme),
            selectinload(NovelSegment.novel),
        )
        .join(Novel, NovelSegment.novel_id == Novel.id)
        .filter(NovelSegment.id == segment_id)
        .first()
    )


def query_prev_segment_id(
    db: Session, novel_id: int, macro_block_id: int, segment_id: int
) -> Optional[int]:
    """Previous segment in the same novel by (macro_block_id, id) order."""
    row = (
        db.query(NovelSegment.id)
        .filter(
            NovelSegment.novel_id == novel_id,
            (NovelSegment.macro_block_id < macro_block_id)
            | (
                (NovelSegment.macro_block_id == macro_block_id)
                & (NovelSegment.id < segment_id)
            ),
        )
        .order_by(desc(NovelSegment.macro_block_id), desc(NovelSegment.id))
        .limit(1)
        .first()
    )
    return row.id if row else None


def query_next_segment_id(
    db: Session, novel_id: int, macro_block_id: int, segment_id: int
) -> Optional[int]:
    """Next segment in the same novel by (macro_block_id, id) order."""
    row = (
        db.query(NovelSegment.id)
        .filter(
            NovelSegment.novel_id == novel_id,
            (NovelSegment.macro_block_id > macro_block_id)
            | (
                (NovelSegment.macro_block_id == macro_block_id)
                & (NovelSegment.id > segment_id)
            ),
        )
        .order_by(NovelSegment.macro_block_id, NovelSegment.id)
        .limit(1)
        .first()
    )
    return row.id if row else None


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
    db: Session, segment_id: int, source: NovelSegment, limit: int
) -> Sequence[SimilarRow]:
    """
    Query similar segments by segment id and source segment.
    Excludes the novel of the source segment.
    """
    embedding = source.embedding

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
        .filter(NovelSegment.id != segment_id, NovelSegment.novel_id != source.novel_id)
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


def get_or_create_characters(
    db: Session, novel_id: int, character_names: list[str]
) -> list[NovelCharacter]:
    """Resolve character names to NovelCharacter objects, creating rows as needed."""
    result: list[NovelCharacter] = []
    for name in character_names:
        name = name.strip()
        if not name:
            continue

        char = (
            db.query(NovelCharacter)
            .filter(
                NovelCharacter.novel_id == novel_id,
                NovelCharacter.name == name,
            )
            .first()
        )
        if char is None:
            char = NovelCharacter(novel_id=novel_id, name=name)
            db.add(char)
            db.flush()

        result.append(char)

    return result


def get_or_create_place(db: Session, novel_id: int, place_name: str) -> NovelPlace:
    """Resolve place name to NovelPlace, creating if needed. Uses 'unknown' for empty or 'unknown'."""
    name = place_name.strip()
    if not name or name.lower() == "unknown":
        name = "unknown"

    place = (
        db.query(NovelPlace)
        .filter(NovelPlace.novel_id == novel_id, NovelPlace.name == name)
        .first()
    )
    if place is None:
        place = NovelPlace(novel_id=novel_id, name=name)
        db.add(place)
        db.flush()
    return place


def get_or_create_mood(db: Session, novel_id: int, mood_name: str) -> NovelMood:
    """Resolve mood name to NovelMood, creating if needed. Uses 'unknown' for empty or 'unknown'."""
    name = mood_name.strip()
    if not name or name.lower() == "unknown":
        name = "unknown"

    mood = (
        db.query(NovelMood)
        .filter(NovelMood.novel_id == novel_id, NovelMood.name == name)
        .first()
    )
    if mood is None:
        mood = NovelMood(novel_id=novel_id, name=name)
        db.add(mood)
        db.flush()
    return mood


def get_or_create_theme(db: Session, novel_id: int, theme_name: str) -> NovelTheme:
    """Resolve theme label to NovelTheme, creating if needed."""
    name = theme_name.strip()
    if not name:
        name = "unknown"

    theme = (
        db.query(NovelTheme)
        .filter(NovelTheme.novel_id == novel_id, NovelTheme.name == name)
        .first()
    )
    if theme is None:
        theme = NovelTheme(novel_id=novel_id, name=name)
        db.add(theme)
        db.flush()
    return theme


def sync_segment_themes(
    db: Session,
    segment: NovelSegment,
    theme_annotations: list[ThemeAnnotation],
) -> None:
    """Replace segment_themes rows for this segment from LLM annotations."""
    db.query(SegmentTheme).filter(SegmentTheme.segment_id == segment.id).delete(
        synchronize_session=False
    )

    for annotation in theme_annotations:
        theme = get_or_create_theme(
            db=db, novel_id=segment.novel_id, theme_name=annotation.name
        )
        db.add(
            SegmentTheme(
                segment_id=segment.id,
                theme_id=theme.id,
                intensity=annotation.intensity,
                tone=annotation.tone,
                manifestation=annotation.manifestation,
            )
        )
    db.flush()


def get_novel_character_names(db: Session, novel_id: int) -> list[str]:
    novel_characters = (
        db.query(NovelCharacter).filter(NovelCharacter.novel_id == novel_id).all()
    )
    return [character.name for character in novel_characters]
