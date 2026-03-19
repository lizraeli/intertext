from pydantic import BaseModel, Field


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
    place: str = Field(
        description="The physical place where this passage occurs. Return 'unknown' if not stated."
    )
    mood: str = Field(
        description="A single word describing the emotional tone (e.g., 'melancholic', 'tense')."
    )
