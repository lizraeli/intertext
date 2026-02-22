from pydantic import BaseModel, Field
from typing import List, Optional


# --- Ingestion Schemas (For OpenAI) ---
class ChunkMetadata(BaseModel):
    primary_themes: list[str] = Field(
        description="Top 2 to 3 overarching literary themes (e.g., 'isolation', 'revenge')."
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


# --- API Endpoints Schemas (For FastAPI) ---
class SegmentResponse(BaseModel):
    id: int
    content: str
    themes: list[str]

    class Config:
        from_attributes = True


class TraversalRequest(BaseModel):
    current_vector: list[float]
    theme_filter: Optional[str] = None
    limit: int = 5


class TraversalResponse(BaseModel):
    novel_title: str
    author: str
    content: str
    similarity_score: float
