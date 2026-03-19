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
from queries import get_novel_character_names, get_or_create_characters
from schemas import ChunkMetadata
from scripts.chunkers import recursive_chunker
from scripts.utils import BookData, get_chapter_blocks, parse_book_from_markdown

load_dotenv()

# Initialize OpenAI Client (automatically picks up OPENAI_API_KEY from the environment)
client = OpenAI()

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

EMBEDDING_MAX_TOKENS = 8191


def extract_chunk_metadata(
    chunk_text: str,
    book: BookData,
    known_character_names: list[str] | None = None,
) -> ChunkMetadata:
    """Extracts metadata from a chunk of text."""
    character_instructions = (
        f'Known characters already identified in this novel (use these exact names when the passage refers to them): {", ".join(known_character_names)}\n\n'
        if known_character_names
        else ""
    )
    character_instructions += """For character names: Use the fullest canonical form you know for each character (e.g., "Jane Eyre" not "Jane" or "the narrator"; "Georgiana Reed" not "Georgiana"). Do not treat nicknames, titles, or narrator references as separate characters—consolidate them into the single canonical name for that person."""
    if known_character_names:
        character_instructions += " For any new character not in the known list, use the fullest canonical form. Do not create duplicates—if a character is in the known list, use that exact name even when the text uses a nickname."

    system_content = f"""You are an expert literary analyst. Extract the requested metadata from this book excerpt.

This excerpt is from "{book.title}" by {book.author}.

{character_instructions}"""
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

            chunks = recursive_chunker.chunk(chapter_data.text)

            print(f"     Generated {len(chunks)} segments.")

            for chunk in chunks:
                known_character_names = get_novel_character_names(
                    db=db, novel_id=novel_record.id
                )
                chunk_metadata = extract_chunk_metadata(
                    chunk_text=chunk.text,
                    book=book,
                    known_character_names=known_character_names,
                )

                characters = get_or_create_characters(
                    db=db,
                    novel_id=novel_record.id,
                    character_names=chunk_metadata.characters,
                )
                metadata_json = chunk_metadata.model_dump()
                metadata_json["chapter"] = chapter_data.chapter
                embedding_vector = openai_embeddings.embed(chunk.text)

                segment = NovelSegment(
                    novel_id=novel_record.id,
                    macro_block_id=block_index,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                    content=chunk.text.strip(),
                    token_count=chunk.token_count,
                    metadata_col=metadata_json,
                    embedding=embedding_vector,
                )
                db.add(segment)
                db.flush()
                segment.characters = characters
                total_segments_processed += 1

                db.commit()
                print(f"     Committed segment {segment.id} to database.")

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
