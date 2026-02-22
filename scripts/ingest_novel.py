import sys
import os
import re
import time
from typing import List
from dotenv import load_dotenv

from openai import OpenAI
from chonkie import LateChunker

from database import SessionLocal
from models import Novel, NovelSegment
from schemas import ChunkMetadata

# 1. Load Environment Variables (.env)
load_dotenv()

# Initialize OpenAI Client (automatically picks up OPENAI_API_KEY from the environment)
client = OpenAI()

# ==========================================
# 2. PROCESSING PIPELINE FUNCTIONS
# ==========================================


def clean_text(text: str) -> str:
    # 1. Replace single newlines with a space (removes hard wrapping)
    # (?<!\n) means "not preceded by a newline"
    # (?!\n) means "not followed by a newline"
    cleaned_text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 2. Collapse multiple spaces into a single space just to be tidy
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return cleaned_text


def get_chapter_blocks(
    raw_text: str, max_words: int = 5000, overlap_words: int = 250
) -> list[dict]:
    """
    Attempts to split the novel by chapter headings. If no chapters are found,
    it automatically falls back to sliding-window macro-chunking.
    """
    # Relaxed Regex: allows leading spaces/BOMs (\s*), makes it case-insensitive,
    # and doesn't strictly demand a newline at the end.
    pattern = re.compile(
        r"^\s*(CHAPTER\s+[A-Z0-9IVXLCM]+)", re.MULTILINE | re.IGNORECASE
    )
    matches = list(pattern.finditer(raw_text))
    blocks = []

    # ==========================================
    # THE FALLBACK: No chapters found
    # ==========================================
    if not matches:
        print(
            "     ⚠️ No standard chapters found. Falling back to sliding-window macro-chunking."
        )
        words = raw_text.split()
        step_size = max_words - overlap_words

        for i in range(0, len(words), step_size):
            window = words[i : i + max_words]
            blocks.append({"chapter": "Continuous Text", "text": " ".join(window)})
        return blocks

    # ==========================================
    # THE PRIMARY: Chapters were found
    # ==========================================
    for i, match in enumerate(matches):
        # We use group(1) to grab just the "CHAPTER X" part, ignoring any matched spaces
        chapter_title = match.group(1).strip()

        # The text starts after the entire matched line
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

        chapter_text = raw_text[start_idx:end_idx].strip()

        # Skip empty chapters (sometimes caused by weird formatting)
        if not chapter_text:
            continue

        blocks.append({"chapter": chapter_title, "text": clean_text(chapter_text)})

    return blocks


def create_macro_chunks(
    raw_text: str, max_words: int = 5000, overlap_words: int = 250
) -> List[str]:
    """Slices a massive text into overlapping macro blocks for the embedding model."""
    # --- TEXT CLEANING STEP ---

    # 1. Replace single newlines with a space (removes hard wrapping)
    # (?<!\n) means "not preceded by a newline"
    # (?!\n) means "not followed by a newline"
    cleaned_text = re.sub(r"(?<!\n)\n(?!\n)", " ", raw_text)

    # 2. Collapse multiple spaces into a single space just to be tidy
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    # ----------------------------------

    words = cleaned_text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words - overlap_words)
    ]


def extract_metadata(chunk_text: str) -> dict:
    """Passes text to OpenAI gpt-4o-mini to extract structured JSON metadata."""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert literary analyst. Extract the requested metadata from this book excerpt.",
            },
            {"role": "user", "content": chunk_text},
        ],
        response_format=ChunkMetadata,
        temperature=0.1,
    )
    return completion.choices[0].message.parsed.model_dump()


# ==========================================
# 3. MAIN INGESTION ORCHESTRATOR
# ==========================================


def ingest_book_to_supabase(file_path: str, title: str, author: str, year: int):
    # 1. Read the raw text
    print(f"Reading '{title}'...")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 2. Initialize the AI Models
    print("Loading Nomic Embedding Model & Chonkie...")
    chunker = LateChunker(
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        chunk_size=512,
        min_characters_per_chunk=50,
        trust_remote_code=True,
    )

    # 3. Create the Database Session
    db = SessionLocal()
    try:
        # Check if novel already exists to prevent duplication
        existing_novel = db.query(Novel).filter(Novel.title == title).first()
        if existing_novel:
            print(
                f"Novel '{title}' already exists in the database. Aborting to prevent duplicates."
            )
            return

        # Create the Parent Record
        novel_record = Novel(title=title, author=author, publication_year=year)
        db.add(novel_record)
        db.flush()  # Gets the novel_record.id without committing the transaction yet

        # 4. Process the Text
        chapter_blocks = get_chapter_blocks(full_text)
        print(
            f"Generated {len(chapter_blocks)} chapter-blocks. Beginning processing pipeline..."
        )

        total_segments_processed = 0
        start_time = time.time()

        for block_index, chapter_data in enumerate(chapter_blocks):
            print(f"  -> Processing Block {block_index + 1}/{len(chapter_blocks)}...")

            # Chonkie performs the Late Chunking math
            segments = chunker(chapter_data["text"])

            for segment in segments:
                # Ask OpenAI for the themes, mood, characters, and setting
                metadata_json = extract_metadata(segment.text)
                metadata_json["chapter"] = chapter_data["chapter"]

                # Create the SQLAlchemy object
                db_seg = NovelSegment(
                    novel_id=novel_record.id,
                    macro_block_id=block_index,
                    content=segment.text.strip(),
                    token_count=segment.token_count,
                    metadata_col=metadata_json,
                    embedding=segment.embedding.tolist(),
                )
                total_segments_processed += 1

                # Add the segments for this specific block
                db.add(db_seg)

                # Commit immediately to flush them to Supabase and clear local RAM
                db.commit()
                print(f"     Committed segment {db_seg.id} to database.")

        elapsed = time.time() - start_time
        print(f"\nSuccess! '{title}' ingested.")
        print(
            f"Total Segments: {total_segments_processed} | Time: {elapsed:.2f} seconds."
        )

    except Exception as e:
        db.rollback()
        print(f"An error occurred during ingestion: {e}")
    finally:
        db.close()


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    ingest_book_to_supabase(
        "books/jane_eyre.txt", "Jane Eyre", "Charlotte Brontë", 1847
    )
