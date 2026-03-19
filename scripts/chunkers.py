from chonkie import (
    RecursiveLevel,
    RecursiveRules,
    RecursiveChunker,
)

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
