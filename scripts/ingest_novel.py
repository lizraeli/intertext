import re
import time
from typing import List
from dotenv import load_dotenv
import tiktoken

from openai import OpenAI
from chonkie import SemanticChunker
from chonkie.embeddings import OpenAIEmbeddings


from sqlalchemy import func

from sqlalchemy.orm import Session

from database import SessionLocal
from models import Novel, NovelCharacter, NovelSegment
from schemas import ChunkMetadata
from scripts.chunkers import recursive_chunker
from scripts.utils import BookData, get_chapter_blocks, parse_book_from_markdown

load_dotenv()

# Initialize OpenAI Client (automatically picks up OPENAI_API_KEY from the environment)
client = OpenAI()

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

EMBEDDING_MAX_TOKENS = 8191


def extract_chunk_metadata(chunk_text: str) -> ChunkMetadata:
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
        temperature=0.4,
    )
    return completion.choices[0].message.parsed


def get_or_create_character_ids(
    db: Session, novel_id: int, character_names: list[str]
) -> list[int]:
    """Resolve character names to IDs, creating NovelCharacter rows as needed."""
    ids = []
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

        ids.append(char.id)

    return ids


def ingest_book_to_db(book: BookData):
    print(f"Ingesting '{book.title}'...")

    db = SessionLocal()
    try:
        # Find or create the novel record to support resuming
        novel_record = db.query(Novel).filter(Novel.title == book.title).first()
        resume_from_block = 0

        if novel_record:
            max_block = (
                db.query(func.max(NovelSegment.macro_block_id))
                .filter(NovelSegment.novel_id == novel_record.id)
                .scalar()
            )

            if max_block is not None:
                # Delete segments from the last block (may be incomplete)
                db.query(NovelSegment).filter(
                    NovelSegment.novel_id == novel_record.id,
                    NovelSegment.macro_block_id == max_block,
                ).delete()
                db.commit()
                resume_from_block = max_block
                print(f"Resuming from block {resume_from_block} (0-indexed)...")
            else:
                print(
                    f"Novel '{book.title}' exists but has no segments. Starting fresh..."
                )
        else:
            novel_record = Novel(
                title=book.title, author=book.author, publication_year=book.year
            )
            db.add(novel_record)
            db.flush()
            db.commit()

        # 4. Process the Text
        chapter_blocks = get_chapter_blocks(book.body)
        print(
            f"Generated {len(chapter_blocks)} chapter-blocks. Beginning processing pipeline..."
        )

        total_segments_processed = 0
        start_time = time.time()

        for block_index, chapter_data in enumerate(chapter_blocks):
            if block_index < resume_from_block:
                print(
                    f"  -> Skipping Chapter {chapter_data.chapter}/{len(chapter_blocks)} (already ingested)"
                )
                continue

            print(
                f"  -> Processing Chapter {chapter_data.chapter}/{len(chapter_blocks)}..."
            )

            segments = recursive_chunker.chunk(chapter_data.text)

            print(f"     Generated {len(segments)} segments.")

            for segment in segments:
                chunk_metadata = extract_chunk_metadata(segment.text)

                character_ids = get_or_create_character_ids(
                    db=db,
                    novel_id=novel_record.id,
                    character_names=chunk_metadata.characters,
                )
                metadata_json = chunk_metadata.model_dump()
                metadata_json["chapter"] = chapter_data.chapter
                embedding_vector = openai_embeddings.embed(segment.text)

                db_seg = NovelSegment(
                    novel_id=novel_record.id,
                    macro_block_id=block_index,
                    start_index=segment.start_index,
                    end_index=segment.end_index,
                    content=segment.text.strip(),
                    token_count=segment.token_count,
                    metadata_col=metadata_json,
                    character_ids=character_ids,
                    embedding=embedding_vector,
                )
                total_segments_processed += 1

                db.add(db_seg)
                db.commit()
                print(f"     Committed segment {db_seg.id} to database.")

        elapsed = time.time() - start_time
        print(f"\nSuccess! '{book.title}' ingested.")
        print(
            f"Total Segments: {total_segments_processed} | Time: {elapsed:.2f} seconds."
        )

    except Exception as e:
        db.rollback()
        print(f"An error occurred during ingestion: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    import json

    with open("scripts/ingest_novel_inputs.json", "r", encoding="utf-8") as f:
        inputs = json.load(f)

    file_path = inputs["file_path"]
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    book = parse_book_from_markdown(full_text)
    ingest_book_to_db(book)
