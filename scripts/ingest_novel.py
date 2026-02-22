import sys
import os
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


def create_macro_chunks(
    raw_text: str, max_words: int = 5000, overlap_words: int = 250
) -> List[str]:
    """Slices a massive text into overlapping macro blocks for the embedding model."""
    words = raw_text.split()
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
        macro_blocks = create_macro_chunks(full_text)
        print(
            f"Generated {len(macro_blocks)} macro-blocks. Beginning processing pipeline..."
        )

        total_segments_processed = 0
        start_time = time.time()

        for block_index, macro_text in enumerate(macro_blocks):
            print(f"  -> Processing Block {block_index + 1}/{len(macro_blocks)}...")

            # Chonkie performs the Late Chunking math
            segments = chunker(macro_text)

            db_segments_to_insert = []
            for segment in segments:
                # Ask OpenAI for the themes, mood, characters, and setting
                metadata_json = extract_metadata(segment.text)

                # Create the SQLAlchemy object
                db_seg = NovelSegment(
                    novel_id=novel_record.id,
                    macro_block_id=block_index,
                    content=segment.text.strip(),
                    token_count=segment.token_count,
                    metadata_col=metadata_json,
                    embedding=segment.embedding.tolist(),
                )
                db_segments_to_insert.append(db_seg)
                total_segments_processed += 1

                # Add the segments for this specific block
                db.add_all(db_segments_to_insert)

                # Commit immediately to flush them to Supabase and clear local RAM
                db.commit()
                print(
                    f"     Committed {len(db_segments_to_insert)} segments to database."
                )

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
