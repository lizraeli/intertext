import re
import time
from typing import List
from dotenv import load_dotenv

from openai import OpenAI
from chonkie import SemanticChunker
from chonkie.embeddings import OpenAIEmbeddings


from sqlalchemy import func

from database import SessionLocal
from models import Novel, NovelSegment
from schemas import ChunkMetadata

# 1. Load Environment Variables (.env)
load_dotenv()

# Initialize OpenAI Client (automatically picks up OPENAI_API_KEY from the environment)
client = OpenAI()

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


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
    pattern = re.compile(
        r"^\s*(?:#{1,3}\s+)?(CHAPTER\s+[A-Z0-9IVXLCM]+[^\n]*)",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(pattern.finditer(raw_text))
    blocks = []

    # Fallback: No chapters found

    if not matches:
        return [
            {"chapter": "Continuous Text", "text": chunk}
            for chunk in create_macro_chunks(
                raw_text, max_words=max_words, overlap_words=overlap_words
            )
        ]

    # Primary: Chapters were found

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


def _is_sentence_end(token: str) -> bool:
    word = token.rstrip()
    if not word or word[-1] not in ".!?\"'":
        return False
    core = word.rstrip("\"'\u2018\u2019\u201c\u201d")
    if core.lower() in _ABBREVIATIONS:
        return False
    return True


def create_macro_chunks(
    text: str, max_words: int = 5000, overlap_words: int = 250
) -> List[str]:
    """Slices a massive text into overlapping macro blocks, splitting at sentence boundaries."""
    tokens = re.findall(r"\S+\s*", text)
    step = max_words - overlap_words
    chunks = []

    for i in range(0, len(tokens), step):
        end = min(i + max_words, len(tokens))

        if end < len(tokens):
            for j in range(end - 1, max(i, end - 200) - 1, -1):
                if _is_sentence_end(tokens[j]):
                    end = j + 1
                    break

        chunks.append("".join(tokens[i:end]).strip())

    return chunks


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


def ingest_book_to_supabase(file_path: str, title: str, author: str, year: int):
    print(f"Reading '{title}'...")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunker = SemanticChunker(
        embedding_model=openai_embeddings,
        chunk_size=512,
        min_sentences_per_chunk=3,
        similarity_window=8,
        filter_tolerance=0.4,
        threshold=0.99,
        delim=["\n\n"],
    )

    db = SessionLocal()
    try:
        # Find or create the novel record to support resuming
        novel_record = db.query(Novel).filter(Novel.title == title).first()
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
                print(f"Novel '{title}' exists but has no segments. Starting fresh...")
        else:
            novel_record = Novel(title=title, author=author, publication_year=year)
            db.add(novel_record)
            db.flush()
            db.commit()

        # 4. Process the Text
        chapter_blocks = get_chapter_blocks(full_text)
        print(
            f"Generated {len(chapter_blocks)} chapter-blocks. Beginning processing pipeline..."
        )

        total_segments_processed = 0
        start_time = time.time()

        for block_index, chapter_data in enumerate(chapter_blocks):
            if block_index < resume_from_block:
                print(
                    f"  -> Skipping Chapter {chapter_data["chapter"]}/{len(chapter_blocks)} (already ingested)"
                )
                continue

            print(
                f"  -> Processing Chapter {chapter_data["chapter"]}/{len(chapter_blocks)}..."
            )

            segments = chunker(chapter_data["text"])

            print(f"     Generated {len(segments)} semantic segments.")

            for segment in segments:
                metadata_json = extract_metadata(segment.text)
                metadata_json["chapter"] = chapter_data["chapter"]

                embedding_vector = openai_embeddings.embed(segment.text)

                db_seg = NovelSegment(
                    novel_id=novel_record.id,
                    macro_block_id=block_index,
                    content=segment.text.strip(),
                    token_count=segment.token_count,
                    metadata_col=metadata_json,
                    embedding=embedding_vector,
                )
                total_segments_processed += 1

                db.add(db_seg)
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


# 4. EXECUTION
if __name__ == "__main__":
    import json

    with open("scripts/ingest_novel_inputs.json", "r") as f:
        inputs = json.load(f)

    ingest_book_to_supabase(
        inputs["file_path"], inputs["title"], inputs["author"], inputs["year"]
    )
