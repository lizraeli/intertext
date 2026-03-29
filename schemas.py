import re

from pydantic import BaseModel
from typing import Optional, cast

from models import Novel, NovelChapter, NovelSegment, SegmentTheme
from queries import RandomSegmentsRow, SimilarRow


class NovelResponse(BaseModel):
    @staticmethod
    def from_row(novel: Novel) -> "NovelResponse":
        return NovelResponse(
            id=novel.id,
            title=novel.title,
            author=novel.author,
            publication_year=novel.publication_year,
        )

    id: int
    title: str
    author: str
    publication_year: Optional[int]


class ChapterResponse(BaseModel):
    @staticmethod
    def from_row(chapter: NovelChapter) -> "ChapterResponse":
        return ChapterResponse(
            id=chapter.id,
            title=chapter.title,
            block_index=chapter.block_index,
        )

    id: int
    title: str
    block_index: int


class ChapterDetailResponse(BaseModel):
    @staticmethod
    def from_row(chapter: NovelChapter) -> "ChapterDetailResponse":
        segments = chapter.segments
        first_segment = cast(
            NovelSegment,
            min(segments, key=lambda s: s.start_index) if segments else None,
        )

        characters: list[str] = sorted(
            {c.name for seg in segments for c in seg.characters}
        )
        places: list[str] = sorted(
            {seg.place.name for seg in segments if seg.place.name != "unknown"}
        )

        return ChapterDetailResponse(
            id=chapter.id,
            title=chapter.title,
            block_index=chapter.block_index,
            opening_line=(
                extract_opening_line(first_segment.content) if first_segment else ""
            ),
            first_segment_id=first_segment.id if first_segment else None,
            characters=characters,
            places=places,
        )

    id: int
    title: str
    block_index: int
    opening_line: str
    first_segment_id: Optional[int]
    characters: list[str]
    places: list[str]


class NovelChaptersResponse(BaseModel):
    novel_title: str
    author: str
    publication_year: Optional[int]
    chapters: list[ChapterDetailResponse]


class SegmentThemeResponse(BaseModel):
    @staticmethod
    def from_row(row: SegmentTheme) -> "SegmentThemeResponse":
        return SegmentThemeResponse(
            name=row.theme.name,
            intensity=row.intensity,
            tone=row.tone,
            manifestation=row.manifestation,
        )

    name: str
    intensity: float
    tone: float
    manifestation: str


class SegmentResponse(BaseModel):
    @staticmethod
    def from_row(row: NovelSegment) -> "SegmentResponse":
        return SegmentResponse(
            id=row.id,
            content=row.content,
            themes=[SegmentThemeResponse.from_row(theme) for theme in row.themes],
            characters=[character.name for character in row.characters],
            place=row.place.name,
            mood=row.mood.name,
            chapter_id=row.chapter.id,
            chapter_title=row.chapter.title,
        )

    id: int
    content: str
    themes: list[SegmentThemeResponse]
    characters: list[str]
    place: str
    mood: str
    chapter_id: int
    chapter_title: str

    class Config:
        from_attributes = True


class TraversalRequest(BaseModel):
    current_embedding: list[float]
    theme_filter: Optional[str] = None
    limit: int = 5


class TraversalResponse(BaseModel):
    novel_title: str
    author: str
    content: str
    similarity_score: float


# --- Frontend API Schemas ---

_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "st.",
        "prof.",
        "rev.",
        "sr.",
        "jr.",
        "etc.",
        "vs.",
        "vol.",
        "no.",
        "gen.",
        "col.",
        "lt.",
        "sgt.",
        "capt.",
        "govt.",
        "approx.",
        "fig.",
        "inc.",
        "ltd.",
        "dept.",
    }
)


def extract_opening_line(content: str, max_chars: int = 200) -> str:
    """Extract the first sentence from segment content, respecting abbreviations."""
    text = content[: max_chars * 2].replace("\n", " ").strip()
    for match in re.finditer(r"[.!?][\"'\u2019\u201d]*\s", text):
        end = match.end()
        before_punct = text[: match.start()].split()
        if before_punct and before_punct[-1].lower() in _ABBREVIATIONS:
            continue
        line = text[:end].strip()
        if len(line) >= 20:
            return line
    return text[:max_chars].strip()


class SegmentPreview(BaseModel):
    @staticmethod
    def from_row(row: RandomSegmentsRow) -> "SegmentPreview":
        return SegmentPreview(
            id=row.id,
            opening_line=extract_opening_line(row.content),
            mood=row.metadata_col.get("mood", "unknown"),
        )

    id: int
    opening_line: str
    mood: str


class SimilarSegmentPreview(BaseModel):
    @staticmethod
    def from_row(row: SimilarRow) -> "SimilarSegmentPreview":
        return SimilarSegmentPreview(
            id=row.id,
            opening_line=extract_opening_line(row.content),
            mood=row.metadata_col.get("mood", "unknown"),
            novel_title=row.title,
            author=row.author,
            similarity_score=1.0 - row.distance,
        )

    id: int
    opening_line: str
    mood: str
    novel_title: str
    author: str
    similarity_score: float


class FullSegmentResponse(BaseModel):
    @staticmethod
    def from_row(
        row: NovelSegment,
        segment_index: int,
        chapter_segment_count: int,
        prev_segment_id: Optional[int] = None,
        next_segment_id: Optional[int] = None,
    ) -> "FullSegmentResponse":
        novel = row.novel

        return FullSegmentResponse(
            id=row.id,
            content=row.content,
            novel_id=novel.id,
            novel_title=novel.title,
            author=novel.author,
            year=novel.publication_year,
            mood=row.mood.name,
            themes=[SegmentThemeResponse.from_row(theme) for theme in row.themes],
            place=row.place.name,
            characters=[character.name for character in row.characters],
            chapter_id=row.chapter.id,
            chapter_title=row.chapter.title,
            prev_segment_id=prev_segment_id,
            next_segment_id=next_segment_id,
            segment_index=segment_index,
            chapter_segment_count=chapter_segment_count,
        )

    id: int
    novel_id: int
    content: str
    novel_title: str
    author: str
    year: Optional[int]
    mood: str
    themes: list[SegmentThemeResponse]
    characters: list[str]
    place: str
    chapter_id: int
    chapter_title: str
    prev_segment_id: Optional[int] = None
    next_segment_id: Optional[int] = None
    segment_index: int = 1
    chapter_segment_count: int = 1
