"""
Align chapter audio to segment boundaries using faster-whisper.

Usage:
    bash scripts/align_audio.sh --novel-id 78
    bash scripts/align_audio.sh --novel-id 78 --chapter 0
    bash scripts/align_audio.sh --novel-id 78 --force
"""

import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from sqlalchemy.orm import Session

from database import SessionLocal
from models import NovelChapter, NovelSegment, SegmentAudio
from queries import query_novel_by_id, query_chapters_for_novel
from schemas import WordTiming

load_dotenv()

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


class ManifestChapter(TypedDict):
    block_index: int
    title: str
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
    word_timings: list[WordTiming | None]


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


def normalize_text(text: str) -> str:
    """Normalize text for comparison: collapse whitespace, normalize unicode."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " -- ").replace("\u2013", " - ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_match(word: str) -> str:
    """Strip punctuation and lowercase for fuzzy comparison."""
    return re.sub(r"[^\w]", "", word.lower())


@dataclass
class TranscribedWord:
    word: str
    start_ms: int
    end_ms: int
    confidence: float


def align_words_to_content(
    content: str, transcribed_words: list[TranscribedWord]
) -> list[WordTiming | None]:
    """
    Greedy fuzzy-align transcribed words to whitespace-split content words.

    Returns a positional array with one entry per content word: a WordTiming
    if matched, or None if the content word had no audio counterpart.
    """
    normalized_content = normalize_text(content)
    content_words = normalized_content.split()
    # Create a list of None values for each content word
    result: list[WordTiming | None] = [None] * len(content_words)

    transcribed_idx = 0
    for content_idx, content_word in enumerate(content_words):
        if transcribed_idx >= len(transcribed_words):
            break

        content_normalized_word = normalize_for_match(content_word)
        if not content_normalized_word:
            continue

        transcribed_normalized_word = normalize_for_match(
            transcribed_words[transcribed_idx].word
        )

        # Match if equal or if one is a prefix of the other
        if (
            content_normalized_word == transcribed_normalized_word
            or content_normalized_word.startswith(transcribed_normalized_word)
            or transcribed_normalized_word.startswith(content_normalized_word)
        ):
            matched_word = transcribed_words[transcribed_idx]
            result[content_idx] = WordTiming(
                start_ms=matched_word.start_ms, end_ms=matched_word.end_ms
            )
            transcribed_idx += 1
            continue

        # Search ahead in transcribed words to resync after drift
        max_lookahead = min(6, len(transcribed_words) - transcribed_idx)
        for skip in range(1, max_lookahead):
            candidate = normalize_for_match(transcribed_words[transcribed_idx + skip].word)
            if (
                content_normalized_word == candidate
                or content_normalized_word.startswith(candidate)
                or candidate.startswith(content_normalized_word)
            ):
                transcribed_idx += skip
                matched_word = transcribed_words[transcribed_idx]
                result[content_idx] = WordTiming(
                    start_ms=matched_word.start_ms, end_ms=matched_word.end_ms
                )
                transcribed_idx += 1
                break

    return result


def get_ordered_segments(db: Session, chapter_id: int) -> list[NovelSegment]:
    """Fetch segments for a chapter ordered by position."""
    return (
        db.query(NovelSegment)
        .filter(NovelSegment.chapter_id == chapter_id)
        .order_by(NovelSegment.start_index)
        .all()
    )


def map_words_to_segments(
    transcribed_words: list[TranscribedWord], segments: list[NovelSegment]
) -> list[SegmentTiming | None]:
    """
    Align transcribed words to segment content at the full chapter level,
    then split the positional results back into per-segment timings.
    """
    chapter_content = "\n\n".join(seg.content for seg in segments)
    chapter_timings = align_words_to_content(chapter_content, transcribed_words)

    chapter_words = normalize_text(chapter_content).split()
    total_matched = sum(1 for t in chapter_timings if t is not None)
    print(
        f"      Chapter-level alignment: {total_matched}/{len(chapter_words)} "
        f"content words matched ({len(transcribed_words)} transcribed words)"
    )

    results: list[SegmentTiming | None] = []
    offset = 0

    for segment in segments:
        seg_word_count = len(normalize_text(segment.content).split())
        segment_timings = chapter_timings[offset : offset + seg_word_count]
        offset += seg_word_count

        timed = [t for t in segment_timings if t is not None]
        print(
            f"      Segment {segment.id}: {len(timed)}/{seg_word_count} words matched"
        )

        if not timed:
            results.append(None)
            continue

        start_ms = timed[0].start_ms
        end_ms = timed[-1].end_ms

        confidence = len(timed) / seg_word_count if seg_word_count > 0 else 0.0

        results.append(
            SegmentTiming(
                segment_id=segment.id,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=round(confidence, 4),
                word_timings=list(segment_timings),
            )
        )

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
    model: WhisperModel,
    chapter: NovelChapter,
    audio_path: Path,
    audio_key: str,
    force: bool,
):
    """Run alignment for a single chapter."""
    segments = get_ordered_segments(db, chapter.id)
    if not segments:
        print(f"    No segments found, skipping.")
        return

    if not force:
        existing_count = (
            db.query(SegmentAudio)
            .filter(SegmentAudio.segment_id.in_([s.id for s in segments]))
            .count()
        )
        if existing_count == len(segments):
            print(f"    Already aligned ({existing_count} segments), skipping.")
            return

    print(f"    Transcribing {audio_path.name} ({len(segments)} segments)...")
    start = time.time()

    print(f"      Loading audio and running encoder...")
    whisper_segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    print(f"      Decoding with word timestamps...")
    words: list[TranscribedWord] = []
    chunk_count = 0
    for segment in whisper_segments:
        chunk_count += 1
        word_count = len(segment.words) if segment.words else 0
        print(
            f"        Chunk {chunk_count}: {word_count} words ({segment.start:.1f}s - {segment.end:.1f}s)"
        )
        if segment.words:
            for whisper_word in segment.words:
                words.append(
                    TranscribedWord(
                        word=whisper_word.word.strip(),
                        start_ms=int(whisper_word.start * 1000),
                        end_ms=int(whisper_word.end * 1000),
                        confidence=round(float(whisper_word.probability), 4),
                    )
                )

    chapter_duration_ms = int(info.duration * 1000)

    elapsed = time.time() - start
    print(
        f"      Transcription done in {elapsed:.1f}s ({len(words)} words, {chapter_duration_ms}ms duration)"
    )

    print(f"    Aligning words to {len(segments)} segments...")
    timings = map_words_to_segments(words, segments)
    postprocess_boundaries(timings, chapter_duration_ms)

    matched_count = sum(1 for t in timings if t is not None)
    print(f"      Matched {matched_count}/{len(segments)} segments")

    print(f"    Writing to database...")
    aligned_count = 0
    for timing in timings:
        if timing is None:
            continue

        status = (
            "aligned" if timing.confidence >= CONFIDENCE_THRESHOLD else "low_confidence"
        )
        words_json = [
            timing.model_dump() if timing is not None else None
            for timing in timing.word_timings
        ]

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
    model_size: str


def parse_args() -> AlignArgs:
    parser = argparse.ArgumentParser(description="Align chapter audio to segments")
    parser.add_argument("--novel-id", type=int, required=True)
    parser.add_argument(
        "--chapter", type=int, default=None, help="Single block_index to align"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-align already aligned chapters"
    )
    parser.add_argument(
        "--model-size", default="small", help="Whisper model size (default: small)"
    )
    args = parser.parse_args()
    return AlignArgs(
        novel_id=args.novel_id,
        chapter=args.chapter,
        force=args.force,
        model_size=args.model_size,
    )


def main():
    args = parse_args()

    manifest, audio_dir = find_manifest(args.novel_id)
    novel_slug = manifest["novel_slug"]

    print(f"Aligning audio for novel_id={args.novel_id} (slug: {novel_slug})")
    print(f"  Model: {args.model_size}, Force: {args.force}")

    db = SessionLocal()
    try:
        novel, chapter_map = validate_manifest(db, manifest, audio_dir)
        print(f"  Validated: '{novel.title}' ({len(manifest['chapters'])} chapters)")

        print(f"  Loading Whisper model '{args.model_size}'...")
        model = WhisperModel(args.model_size, device="cpu", compute_type="int8")

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

            print(f"\n  Chapter {block_index}: {ch_info['title']}")
            align_chapter(db, model, db_chapter, audio_path, audio_key, args.force)

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
