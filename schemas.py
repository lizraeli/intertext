import re

from pydantic import BaseModel, Field
from typing import Optional

from models import NovelSegment
from queries import RandomSegmentsRow, SimilarRow


# --- Ingestion Schemas (For OpenAI) ---
class ThemeAnnotation(BaseModel):
    name: str = Field(
        description="The theme name (e.g., 'isolation', 'revenge', 'class struggle')."
    )
    intensity: float = Field(
        description="Score from 0.0 to 1.0: 0.1 means faintly present in the background, 1.0 means the overwhelming central concern of the passage."
    )
    tone: float = Field(
        description="Score from -1.0 to 1.0: -1.0 is the darkest or most painful expression of the theme, 0.0 is neutral or ambivalent, 1.0 is the most hopeful or transcendent expression."
    )
    manifestation: str = Field(
        description="Write a single sentence describing how this theme concretely manifests in this specific passage. This sentence must be specific enough that a reader who has not seen the passage can form a vivid visual or emotional impression from it alone. Do NOT write a sentence that could apply to any passage with this theme — do not restate the keyword or write a general observation. Describe what the author actually does: the specific image, action, voice, or detail through which the theme appears."
    )


class ChunkMetadata(BaseModel):
    primary_themes: list[ThemeAnnotation] = Field(
        description="The top 2 to 3 overarching literary themes in this passage."
    )
    characters: list[str] = Field(
        description="Names of characters actively participating or mentioned."
    )
    setting: str = Field(
        description="The physical location, if discernible. Return 'Unknown' if not stated."
    )
    mood: str = Field(
        description="A single word describing the emotional tone (e.g., 'melancholic', 'tense')."
    )


# --- API Endpoints Schemas ---
class SegmentResponse(BaseModel):
    id: int
    content: str
    themes: list[ThemeAnnotation]

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
        prev_segment_id: Optional[int] = None,
        next_segment_id: Optional[int] = None,
    ) -> "FullSegmentResponse":
        metadata = row.metadata_col
        novel = row.novel

        return FullSegmentResponse(
            id=row.id,
            content=row.content,
            novel_id=novel.id,
            novel_title=novel.title,
            author=novel.author,
            year=novel.publication_year,
            mood=metadata.get("mood", "unknown"),
            themes=metadata.get("primary_themes", []),
            setting=metadata.get("setting", "Unknown"),
            characters=[character.name for character in row.characters],
            prev_segment_id=prev_segment_id,
            next_segment_id=next_segment_id,
        )

    id: int
    novel_id: int
    content: str
    novel_title: str
    author: str
    year: Optional[int]
    mood: str
    themes: list[ThemeAnnotation]
    characters: list[str]
    setting: str
    prev_segment_id: Optional[int] = None
    next_segment_id: Optional[int] = None
