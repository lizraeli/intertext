from typing import Any, Optional, Protocol, Sequence

from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql.expression import func
from sqlalchemy import desc

from models import (
    Novel,
    NovelCharacter,
    NovelChapter,
    NovelMood,
    NovelPlace,
    NovelSegment,
    NovelTheme,
    SegmentAudio,
    SegmentTheme,
)
from llm_schemas import ThemeAnnotation


def query_all_novels(db: Session) -> list[Novel]:
    return db.query(Novel).order_by(Novel.title).all()


def query_novel_by_id(db: Session, novel_id: int) -> Novel | None:
    return db.query(Novel).filter(Novel.id == novel_id).first()


def get_novel_by_title(db: Session, title: str) -> Novel | None:
    return db.query(Novel).filter(Novel.title == title).first()


def get_max_chapter_block_index(db: Session, novel_id: int) -> int | None:
    return (
        db.query(func.max(NovelChapter.block_index))
        .filter(NovelChapter.novel_id == novel_id)
        .scalar()
    )


def get_chapter_by_novel_and_block(
    db: Session, novel_id: int, block_index: int
) -> NovelChapter | None:
    return (
        db.query(NovelChapter)
        .filter(
            NovelChapter.novel_id == novel_id,
            NovelChapter.block_index == block_index,
        )
        .first()
    )


def delete_segments_for_chapter(db: Session, chapter_id: int) -> None:
    db.query(NovelSegment).filter(NovelSegment.chapter_id == chapter_id).delete(
        synchronize_session=False
    )


def query_chapters_for_novel(db: Session, novel_id: int) -> list[NovelChapter]:
    return (
        db.query(NovelChapter)
        .filter(NovelChapter.novel_id == novel_id)
        .order_by(NovelChapter.block_index)
        .all()
    )


def query_chapters_with_segments(db: Session, novel_id: int) -> list[NovelChapter]:
    """Load chapters with segments eagerly loaded (including each segment's place)."""
    return (
        db.query(NovelChapter)
        .options(
            selectinload(NovelChapter.segments).selectinload(NovelSegment.place),
        )
        .filter(NovelChapter.novel_id == novel_id)
        .order_by(NovelChapter.block_index)
        .all()
    )


def query_novel_segments(db: Session, novel_id: int):
    return (
        db.query(NovelSegment)
        .options(
            selectinload(NovelSegment.characters),
            selectinload(NovelSegment.place),
            selectinload(NovelSegment.mood),
            selectinload(NovelSegment.chapter),
            selectinload(NovelSegment.themes).selectinload(SegmentTheme.theme),
        )
        .join(NovelChapter, NovelSegment.chapter_id == NovelChapter.id)
        .filter(NovelSegment.novel_id == novel_id)
        .order_by(NovelChapter.block_index, NovelSegment.id)
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
            selectinload(NovelSegment.chapter),
            selectinload(NovelSegment.themes).selectinload(SegmentTheme.theme),
            selectinload(NovelSegment.novel),
            selectinload(NovelSegment.audio),
        )
        .join(Novel, NovelSegment.novel_id == Novel.id)
        .filter(NovelSegment.id == segment_id)
        .first()
    )


def query_prev_segment_id(
    db: Session, novel_id: int, block_index: int, segment_id: int
) -> Optional[int]:
    """Previous segment in the same novel by (chapter block_index, id) order."""
    row = (
        db.query(NovelSegment.id)
        .join(NovelChapter, NovelSegment.chapter_id == NovelChapter.id)
        .filter(
            NovelSegment.novel_id == novel_id,
            (NovelChapter.block_index < block_index)
            | (
                (NovelChapter.block_index == block_index)
                & (NovelSegment.id < segment_id)
            ),
        )
        .order_by(desc(NovelChapter.block_index), desc(NovelSegment.id))
        .limit(1)
        .first()
    )
    return row.id if row else None


def query_next_segment_id(
    db: Session, novel_id: int, block_index: int, segment_id: int
) -> Optional[int]:
    """Next segment in the same novel"""
    row = (
        db.query(NovelSegment.id)
        .join(NovelChapter, NovelSegment.chapter_id == NovelChapter.id)
        .filter(
            NovelSegment.novel_id == novel_id,
            (NovelChapter.block_index > block_index)
            | (
                (NovelChapter.block_index == block_index)
                & (NovelSegment.id > segment_id)
            ),
        )
        .order_by(NovelChapter.block_index, NovelSegment.id)
        .limit(1)
        .first()
    )
    return row.id if row else None


def query_segment_position_in_chapter(
    db: Session, chapter_id: int, segment_id: int
) -> int:
    """Position of this segment within its chapter"""
    count = (
        db.query(func.count(NovelSegment.id))
        .filter(
            NovelSegment.chapter_id == chapter_id,
            NovelSegment.id < segment_id,
        )
        .scalar()
    )
    return count + 1


def query_chapter_segment_count(db: Session, chapter_id: int) -> int:
    """Total number of segments in a chapter."""
    return (
        db.query(func.count(NovelSegment.id))
        .filter(NovelSegment.chapter_id == chapter_id)
        .scalar()
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


def get_or_create_chapter(
    db: Session, novel_id: int, block_index: int, title: str
) -> NovelChapter:
    """Resolve chapter by (novel, block index), creating if needed. Updates title on match."""
    chapter = (
        db.query(NovelChapter)
        .filter(
            NovelChapter.novel_id == novel_id,
            NovelChapter.block_index == block_index,
        )
        .first()
    )
    if chapter is None:
        chapter = NovelChapter(novel_id=novel_id, block_index=block_index, title=title)
        db.add(chapter)
        db.flush()
    else:
        chapter.title = title
    return chapter


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


def get_novel_place_names(db: Session, novel_id: int) -> list[str]:
    novel_places = db.query(NovelPlace).filter(NovelPlace.novel_id == novel_id).all()
    return [place.name for place in novel_places]
