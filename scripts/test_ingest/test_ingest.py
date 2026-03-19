import re
import time
import tiktoken

from dotenv import load_dotenv
from openai import OpenAI
from chonkie import (
    RecursiveLevel,
    RecursiveRules,
    RecursiveChunker,
)
from chonkie.embeddings import OpenAIEmbeddings


from scripts.utils import (
    BookData,
    get_chapter_blocks,
    parse_book_from_markdown,
)

# 1. Load Environment Variables (.env)
load_dotenv()

# Initialize OpenAI Client (automatically picks up OPENAI_API_KEY from the environment)
client = OpenAI()

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# text-embedding-3-large uses cl100k_base; max 8191 tokens
EMBEDDING_MAX_TOKENS = 8191


def get_token_count(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


recursive_chunker = RecursiveChunker(
    chunk_size=1024,
    rules=RecursiveRules(
        levels=[
            RecursiveLevel(
                delimiters=["\n\n"],
            )
        ]
    ),
    min_characters_per_chunk=24,
)


def ingest_book(book: BookData):
    print(f"Ingesting '{book.title}'...")

    with open(f"scripts/test_ingest/test_ingest_output.md", "w", encoding="utf-8") as f:
        pass  # clears the file

    try:
        # 4. Process the Text
        chapter_blocks = get_chapter_blocks(book.body)
        print(
            f"Generated {len(chapter_blocks)} chapter-blocks. Beginning processing pipeline..."
        )

        total_segments_processed = 0
        start_time = time.time()

        for chapter_block in chapter_blocks:
            print(f"Chunking chapter block {chapter_block.chapter}...")
            list_of_chunks = recursive_chunker.chunk(chapter_block.text)
            print(f"Generated {len(list_of_chunks)} chunks.")

            for chunk in list_of_chunks:
                with open(
                    "scripts/test_ingest/test_ingest_output.md", "a", encoding="utf-8"
                ) as out:
                    out.write(f"## segment {chunk.id}\n")
                    out.write("\n")
                    out.write(chunk.text.strip())
                    out.write("\n\n\n")
                    total_segments_processed += 1

        elapsed = time.time() - start_time
        print(f"\nSuccess! '{book.title}' ingested.")
        print(
            f"Total Segments: {total_segments_processed} | Time: {elapsed:.2f} seconds."
        )

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
    finally:
        print("Ingestion completed.")


if __name__ == "__main__":
    with open("scripts/test_ingest/test_ingest_input.md", "r", encoding="utf-8") as f:
        full_text = f.read()

    book = parse_book_from_markdown(full_text)
    ingest_book(book)
