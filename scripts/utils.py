from dataclasses import dataclass
import re
from typing import List, TypeVar, TypedDict


def clean_text(text: str) -> str:
    # 1. Replace single newlines with a space (removes hard wrapping)
    # (?<!\n) means "not preceded by a newline"
    # (?!\n) means "not followed by a newline"
    cleaned_text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 2. Collapse multiple spaces into a single space just to be tidy
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return cleaned_text


@dataclass
class ChapterData:
    chapter: str
    text: str


def get_chapter_blocks(raw_text: str) -> list[ChapterData]:
    """
    Attempts to split the novel by chapter headings.
    Raises a ValueError if no chapters are found.
    """
    pattern = re.compile(
        r"^\s*(#{1,3}\s+[^\n]+)",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(raw_text))
    blocks = []

    if not matches:
        raise ValueError("No chapters found in the text")

    for i, match in enumerate(matches):
        # We use group(1) to grab just the "CHAPTER X" part, ignoring any matched spaces
        chapter_title = match.group(1).strip().lstrip("#").strip()

        # The text starts after the entire matched line
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

        chapter_text = raw_text[start_idx:end_idx].strip()

        # Skip empty chapters (sometimes caused by weird formatting)
        if not chapter_text:
            continue

        blocks.append(ChapterData(chapter=chapter_title, text=clean_text(chapter_text)))

    return blocks


@dataclass
class BookData:
    title: str
    body: str
    author: str
    year: int


def parse_book_from_markdown(full_text: str) -> BookData:
    """
    Parses the title, author, year, and body from the markdown file.
    Raises ValueError if any required header line is missing.
    """
    lines = full_text.split("\n")
    title: str | None = None
    author: str | None = None
    year: int | None = None
    last_metadata_line = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r"^#\s*Title:\s*(.+)$", stripped)
        if m:
            title = m.group(1).strip()
            last_metadata_line = i
            continue
        m = re.match(r"^##\s*Author:\s*(.+)$", stripped)
        if m:
            author = m.group(1).strip()
            last_metadata_line = i
            continue
        m = re.match(r"^##\s*Year:\s*(\d+)$", stripped)
        if m:
            year = int(m.group(1))
            last_metadata_line = i
            continue

    if title is None:
        raise ValueError("Missing required header line: # Title: ...")
    if author is None:
        raise ValueError("Missing required header line: ## Author: ...")
    if year is None:
        raise ValueError("Missing required header line: ## Year: ...")

    # Body starts after the last metadata line; skip optional blank lines and a dash-only line
    body_lines = lines[last_metadata_line + 1 :]
    start = 0
    while start < len(body_lines) and body_lines[start].strip() == "":
        start += 1
    if start < len(body_lines) and re.match(r"^[-]+$", body_lines[start].strip()):
        start += 1
    while start < len(body_lines) and body_lines[start].strip() == "":
        start += 1
    body = "\n".join(body_lines[start:])

    return BookData(title=title, body=body, author=author, year=year)


T = TypeVar("T")


def flatten_list_of_lists(list_of_lists: list[list[T]]) -> list[T]:
    return [item for list in list_of_lists for item in list]
