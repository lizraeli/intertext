from openai import OpenAI

from scripts.utils import BookData
from llm_schemas import ChunkMetadata

# Initialize OpenAI Client (automatically picks up OPENAI_API_KEY from the environment)
client = OpenAI()


_BASE_CHARACTER_INSTRUCTION = (
    "For character names: Use the fullest canonical form you know for each character "
    '(e.g., "Jane Eyre" not "Jane" or "the narrator"; "Georgiana Reed" not "Georgiana"). '
    "Do not treat nicknames, titles, or narrator references as separate characters—"
    "consolidate them into the single canonical name for that person."
    "\n\n"
)

_BASE_PLACE_INSTRUCTION = (
    "For place names: Use the fullest canonical form you know for each place. "
    "Do not create duplicates."
    "\n\n"
)


def extract_chunk_metadata(
    chunk_text: str,
    book: BookData,
    known_character_names: list[str],
    known_place_names: list[str],
) -> ChunkMetadata:
    """Extracts metadata from a chunk of text."""
    system_content = _get_system_content(
        book=book,
        known_character_names=known_character_names,
        known_place_names=known_place_names,
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f'Excerpt from "{book.title}" by {book.author}:\n\n{chunk_text}',
            },
        ],
        response_format=ChunkMetadata,
        temperature=0.4,
    )
    return completion.choices[0].message.parsed


def _get_system_content(
    book: BookData, known_character_names: list[str], known_place_names: list[str]
) -> str:
    if known_character_names:
        character_instructions = (
            f'Known characters already identified (use these exact names when applicable): {", ".join(known_character_names)}.\n\n'
            f"{_BASE_CHARACTER_INSTRUCTION} "
            "For new characters not in this list, use the fullest canonical form. "
            "Do not create duplicates."
        )
    else:
        character_instructions = _BASE_CHARACTER_INSTRUCTION

    if known_place_names:
        place_instructions = f'Known places already identified (use these exact names when applicable): {", ".join(known_place_names)}.\n\n'
    else:
        place_instructions = _BASE_PLACE_INSTRUCTION

    return f"""You are an expert literary analyst. Extract the requested metadata from this book excerpt.

    This excerpt is from "{book.title}" by {book.author}.

    {character_instructions}
    {place_instructions}"""
