"""
Align chapter audio to segment boundaries using forced alignment.

Usage:
    bash scripts/align_audio.sh --novel-id 78
    bash scripts/align_audio.sh --novel-id 78 --chapter 0
    bash scripts/align_audio.sh --novel-id 78 --force
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
    postprocess_results,
    preprocess_text,
)
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from database import SessionLocal
from models import NovelChapter, NovelSegment, SegmentAudio
from queries import query_novel_by_id, query_chapters_for_novel
from schemas import WordTiming

load_dotenv()

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
SAMPLING_FREQ = 16000


class ManifestChapter(TypedDict):
    chapter: int
    block_index: int
    file: str


class Manifest(TypedDict):
    novel_id: int
    novel_slug: str
    novel_title: str
    audio_base_path: str
    chapters: list[ManifestChapter]


@dataclass
class SegmentTiming:
    segment_id: int
    start_ms: int
    end_ms: int
    confidence: float
    word_timings: list[WordTiming]


CONFIDENCE_THRESHOLD = 0.6
PADDING_MS = 100


def find_manifest(novel_id: int) -> tuple[Manifest, Path]:
    """Scan audio directories for a manifest matching the given novel_id."""
    for manifest_path in AUDIO_DIR.glob("*/manifest.json"):
        with open(manifest_path, "r") as f:
            manifest: Manifest = json.load(f)
        if manifest.get("novel_id") == novel_id:
            return manifest, manifest_path.parent
    raise FileNotFoundError(f"No manifest found for novel_id={novel_id} in {AUDIO_DIR}")


@dataclass
class ContentToken:
    word_text: str
    char_start: int
    char_end: int


def tokenize_content(content: str) -> list[ContentToken]:
    """Split content into word tokens with their character offsets.

    Whitespace, em-dashes, and en-dashes are treated as separators
    and do not produce tokens.
    """
    separators = set(" \t\n\r\u2014\u2013")
    tokens: list[ContentToken] = []
    i = 0
    n = len(content)
    while i < n:
        while i < n and content[i] in separators:
            i += 1
        if i >= n:
            break
        start = i
        while i < n and content[i] not in separators:
            i += 1
        tokens.append(ContentToken(content[start:i], start, i))
    return tokens


def clean_word_for_alignment(word: str) -> str:
    """Strip everything except letters and apostrophes for alignment."""
    return re.sub(r"[^a-zA-Z']", "", word).lower()


def segment_has_alignable_words(content: str) -> bool:
    """True if the segment has at least one token that contributes to forced alignment."""
    return any(clean_word_for_alignment(t.word_text) for t in tokenize_content(content))


def align_chapter_text(
    model,
    tokenizer,
    audio_waveform: torch.Tensor,
    chapter_content: str,
) -> list[WordTiming]:
    """Use forced alignment to produce word-level timestamps for chapter content.

    Tokenizes the content, cleans words for the aligner, runs forced alignment,
    and maps the resulting timestamps back to character offsets in the original content.
    """
    print("      Tokenizing content...")
    content_tokens = tokenize_content(chapter_content)

    print("      Cleaning words for alignment...")
    alignable_tokens: list[ContentToken] = []
    clean_words: list[str] = []
    for token in content_tokens:
        cleaned = clean_word_for_alignment(token.word_text)
        if cleaned:
            alignable_tokens.append(token)
            clean_words.append(cleaned)

    if not clean_words:
        return []

    print(
        f"      {len(alignable_tokens)} alignable words "
        f"from {len(content_tokens)} content tokens"
    )

    clean_text = " ".join(clean_words)

    print("      Preprocessing text...")
    tokens_starred, text_starred = preprocess_text(
        clean_text, romanize=True, language="eng"
    )

    print("      Generating emissions...")
    emissions, stride = generate_emissions(model, audio_waveform, batch_size=4)

    print("      Getting alignments...")
    segments, scores, blank_token = get_alignments(
        emissions=emissions, tokens=tokens_starred, tokenizer=tokenizer
    )

    print("      Getting spans...")
    spans = get_spans(tokens=tokens_starred, segments=segments, blank=blank_token)

    print("      Postprocessing results...")
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    print("      Fixing preamble outliers...")
    result: list[WordTiming] = []
    for i, word_ts in enumerate(word_timestamps):
        if i >= len(alignable_tokens):
            break
        token = alignable_tokens[i]
        result.append(
            WordTiming(
                char_start=token.char_start,
                char_end=token.char_end,
                start_ms=int(word_ts["start"] * 1000),
                end_ms=int(word_ts["end"] * 1000),
            )
        )

    print(f"      Aligned {len(result)}/{len(alignable_tokens)} words")
    fix_preamble_outliers(result)
    return result


GAP_THRESHOLD_MS = 2000


def fix_preamble_outliers(timings: list[WordTiming]) -> None:
    """Snap forward any leading words that were misaligned to preamble audio.

    If a word at the start has a gap > GAP_THRESHOLD_MS to the next word,
    it was likely matched to preamble narration (e.g. the book title).
    Snap its timing to just before the next word so playback starts cleanly.
    """
    for i in range(len(timings) - 1):
        gap = timings[i + 1].start_ms - timings[i].end_ms
        if gap <= GAP_THRESHOLD_MS:
            if i > 0:
                print(f"      Fixed {i} preamble outlier(s)")
            break
        timings[i].start_ms = max(0, timings[i + 1].start_ms - 200)
        timings[i].end_ms = timings[i + 1].start_ms


def get_ordered_segments(db: Session, chapter_id: int) -> list[NovelSegment]:
    """Fetch segments for a chapter ordered by position."""
    return (
        db.query(NovelSegment)
        .filter(NovelSegment.chapter_id == chapter_id)
        .order_by(NovelSegment.start_index)
        .all()
    )


def map_words_to_segments(
    chapter_timings: list[WordTiming], segments: list[NovelSegment]
) -> list[SegmentTiming | None]:
    """Split chapter-level word timings into per-segment timings
    with segment-relative character offsets."""
    results: list[SegmentTiming | None] = []
    timing_idx = 0
    seg_char_offset = 0

    for segment in segments:
        seg_length = len(segment.content)
        seg_end = seg_char_offset + seg_length

        segment_word_timings: list[WordTiming] = []
        while (
            timing_idx < len(chapter_timings)
            and chapter_timings[timing_idx].char_start < seg_end
        ):
            timing = chapter_timings[timing_idx]
            segment_word_timings.append(
                WordTiming(
                    char_start=timing.char_start - seg_char_offset,
                    char_end=timing.char_end - seg_char_offset,
                    start_ms=timing.start_ms,
                    end_ms=timing.end_ms,
                )
            )
            timing_idx += 1

        seg_token_count = len(tokenize_content(segment.content))
        matched_count = len(segment_word_timings)
        print(
            f"      Segment {segment.id}: {matched_count}/{seg_token_count} words matched"
        )

        if not segment_word_timings:
            results.append(None)
        else:
            start_ms = segment_word_timings[0].start_ms
            end_ms = segment_word_timings[-1].end_ms
            confidence = matched_count / seg_token_count if seg_token_count > 0 else 0.0

            results.append(
                SegmentTiming(
                    segment_id=segment.id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=round(confidence, 4),
                    word_timings=segment_word_timings,
                )
            )

        seg_char_offset = seg_end + 2  # +2 for the "\n\n" separator

    return results


def postprocess_boundaries(
    segment_timings: list[SegmentTiming | None], chapter_duration_ms: int
):
    """Apply padding, enforce monotonic non-overlap, and clamp to chapter duration."""
    for timing in segment_timings:
        if timing is None:
            continue
        timing.start_ms = max(0, timing.start_ms - PADDING_MS)
        timing.end_ms = min(chapter_duration_ms, timing.end_ms + PADDING_MS)

    prev_end = 0
    for timing in segment_timings:
        if timing is None:
            continue
        if timing.start_ms < prev_end:
            timing.start_ms = prev_end
        if timing.end_ms <= timing.start_ms:
            timing.end_ms = timing.start_ms + 100
        prev_end = timing.end_ms


def align_chapter(
    db: Session,
    model,
    tokenizer,
    chapter: NovelChapter,
    audio_path: Path,
    audio_key: str,
    force: bool,
):
    """Run forced alignment for a single chapter."""
    segments = get_ordered_segments(db, chapter.id)
    if not segments:
        print("    No segments found, skipping.")
        return

    if not force:
        segments_with_words = [
            seg for seg in segments if segment_has_alignable_words(seg.content)
        ]
        if not segments_with_words:
            print("    No alignable words in this chapter, skipping.")
            return

        existing_count = (
            db.query(SegmentAudio)
            .filter(
                SegmentAudio.segment_id.in_([seg.id for seg in segments_with_words])
            )
            .count()
        )

        if existing_count == len(segments_with_words):
            print(
                f"    Already aligned ({existing_count} segments with words), skipping."
            )
            return

    print(f"    Aligning {audio_path.name} ({len(segments)} segments)...")
    start_time = time.time()

    print("      Loading audio...")
    audio_waveform = load_audio(str(audio_path), model.dtype, model.device)
    chapter_duration_ms = int(audio_waveform.size(0) / SAMPLING_FREQ * 1000)

    chapter_content = "\n\n".join(seg.content for seg in segments)
    print(f"      Chapter content: {len(chapter_content)} chars")

    print("      Running forced alignment...")
    chapter_timings = align_chapter_text(
        model, tokenizer, audio_waveform, chapter_content
    )

    elapsed = time.time() - start_time
    print(
        f"      Alignment done in {elapsed:.1f}s "
        f"({len(chapter_timings)} words, {chapter_duration_ms}ms duration)"
    )

    print(f"    Splitting into {len(segments)} segments...")
    timings = map_words_to_segments(chapter_timings, segments)
    postprocess_boundaries(timings, chapter_duration_ms)

    matched_count = sum(1 for t in timings if t is not None)
    print(f"      Matched {matched_count}/{len(segments)} segments")

    print("    Writing to database...")
    aligned_count = 0
    for timing in timings:
        if timing is None:
            continue

        status = (
            "aligned" if timing.confidence >= CONFIDENCE_THRESHOLD else "low_confidence"
        )
        words_json = [wt.model_dump() for wt in timing.word_timings]

        existing = (
            db.query(SegmentAudio)
            .filter(SegmentAudio.segment_id == timing.segment_id)
            .first()
        )
        if existing:
            existing.audio_key = audio_key
            existing.start_ms = timing.start_ms
            existing.end_ms = timing.end_ms
            existing.confidence = timing.confidence
            existing.status = status
            existing.words = words_json
        else:
            db.add(
                SegmentAudio(
                    segment_id=timing.segment_id,
                    audio_key=audio_key,
                    start_ms=timing.start_ms,
                    end_ms=timing.end_ms,
                    confidence=timing.confidence,
                    status=status,
                    words=words_json,
                )
            )
        aligned_count += 1

    db.commit()
    print(f"    Wrote {aligned_count} segment_audio rows.")


def validate_manifest(db: Session, manifest: Manifest, audio_dir: Path):
    """Run pre-alignment validation checks."""
    novel = query_novel_by_id(db, manifest["novel_id"])
    if not novel:
        raise ValueError(f"Novel id={manifest['novel_id']} not found in database.")

    if novel.title != manifest.get("novel_title"):
        print(
            f"  WARNING: DB title '{novel.title}' != manifest title "
            f"'{manifest.get('novel_title')}'"
        )

    db_chapters = query_chapters_for_novel(db, novel.id)
    db_block_indices = {ch.block_index for ch in db_chapters}
    manifest_block_indices = {ch["block_index"] for ch in manifest["chapters"]}

    missing_in_db = manifest_block_indices - db_block_indices
    if missing_in_db:
        raise ValueError(
            f"Manifest block_indices {missing_in_db} not found in DB chapters."
        )

    missing_in_manifest = db_block_indices - manifest_block_indices
    if missing_in_manifest:
        print(f"  WARNING: DB chapters {missing_in_manifest} not listed in manifest.")

    for chapter in manifest["chapters"]:
        audio_path = audio_dir / chapter["file"]
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    return novel, {chapter.block_index: chapter for chapter in db_chapters}


@dataclass
class AlignArgs:
    novel_id: int
    chapter: int | None
    force: bool


def parse_args() -> AlignArgs:
    parser = argparse.ArgumentParser(description="Align chapter audio to segments")
    parser.add_argument("--novel-id", type=int, required=True)
    parser.add_argument(
        "--chapter", type=int, default=None, help="Single block_index to align"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-align already aligned chapters"
    )
    args = parser.parse_args()
    return AlignArgs(
        novel_id=args.novel_id,
        chapter=args.chapter,
        force=args.force,
    )


def main():
    args = parse_args()

    manifest, audio_dir = find_manifest(args.novel_id)
    novel_slug = manifest["novel_slug"]

    print(f"Aligning audio for novel_id={args.novel_id} (slug: {novel_slug})")
    print(f"  Force: {args.force}")

    db = SessionLocal()
    try:
        novel, chapter_map = validate_manifest(db, manifest, audio_dir)
        print(f"  Validated: '{novel.title}' ({len(manifest['chapters'])} chapters)")

        print("  Loading alignment model...")
        model, tokenizer = load_alignment_model(device="cpu", dtype=torch.float32)

        chapters_to_process = manifest["chapters"]
        if args.chapter is not None:
            chapters_to_process = [
                ch for ch in chapters_to_process if ch["block_index"] == args.chapter
            ]
            if not chapters_to_process:
                print(f"  Chapter block_index={args.chapter} not found in manifest.")
                return

        total_start = time.time()
        for ch_info in chapters_to_process:
            block_index = ch_info["block_index"]
            db_chapter = chapter_map[block_index]
            audio_path = audio_dir / ch_info["file"]
            audio_key = f"{novel_slug}/{ch_info['file']}"

            print(f"\n  Chapter {block_index}")
            print(f"    Audio path: {audio_path}")
            align_chapter(
                db, model, tokenizer, db_chapter, audio_path, audio_key, args.force
            )

        total_elapsed = time.time() - total_start
        print(f"\nDone! Total time: {total_elapsed:.1f}s")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
