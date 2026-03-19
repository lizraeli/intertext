import time
from dotenv import load_dotenv
from chonkie.embeddings import OpenAIEmbeddings
from sqlalchemy import func

from database import SessionLocal
from models import Novel, NovelChapter, NovelSegment
from queries import (
    get_novel_character_names,
    get_or_create_characters,
    get_or_create_chapter,
    get_or_create_mood,
    get_or_create_place,
    sync_segment_themes,
)
from scripts.chunkers import recursive_chunker
from scripts.llm import extract_chunk_metadata
from scripts.utils import BookData, get_chapter_blocks, parse_book_from_markdown

load_dotenv()

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

EMBEDDING_MAX_TOKENS = 8191


def ingest_book_to_db(book: BookData):
    print(f"Ingesting '{book.title}'...")

    db = SessionLocal()
    try:
        # Find or create the novel record to support resuming
        novel_record = db.query(Novel).filter(Novel.title == book.title).first()
        resume_from_block = 0

        if novel_record:
            max_block = (
                db.query(func.max(NovelChapter.block_index))
                .filter(NovelChapter.novel_id == novel_record.id)
                .scalar()
            )

            if max_block is not None:
                last_chapter = (
                    db.query(NovelChapter)
                    .filter(
                        NovelChapter.novel_id == novel_record.id,
                        NovelChapter.block_index == max_block,
                    )
                    .first()
                )
                if last_chapter is not None:
                    # Delete segments from the last chapter (may be incomplete)
                    db.query(NovelSegment).filter(
                        NovelSegment.chapter_id == last_chapter.id
                    ).delete(synchronize_session=False)
                db.commit()
                resume_from_block = max_block
                print(f"Resuming from block {resume_from_block} (0-indexed)...")
            else:
                print(
                    f"Novel '{book.title}' exists but has no chapters yet. Starting fresh..."
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

            chapter_row = get_or_create_chapter(
                db=db,
                novel_id=novel_record.id,
                block_index=block_index,
                title=chapter_data.chapter,
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
                place = get_or_create_place(
                    db=db,
                    novel_id=novel_record.id,
                    place_name=chunk_metadata.place,
                )
                mood = get_or_create_mood(
                    db=db,
                    novel_id=novel_record.id,
                    mood_name=chunk_metadata.mood,
                )
                metadata_json = chunk_metadata.model_dump()
                metadata_json["chapter"] = chapter_data.chapter
                embedding_vector = openai_embeddings.embed(chunk.text)

                segment = NovelSegment(
                    novel_id=novel_record.id,
                    chapter_id=chapter_row.id,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                    content=chunk.text.strip(),
                    token_count=chunk.token_count,
                    metadata_col=metadata_json,
                    place_id=place.id,
                    mood_id=mood.id,
                    embedding=embedding_vector,
                )
                db.add(segment)
                db.flush()

                segment.characters = characters

                sync_segment_themes(
                    db=db,
                    segment=segment,
                    theme_annotations=chunk_metadata.primary_themes,
                )
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
