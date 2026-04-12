"""
Quick test of ctc-forced-aligner on chapter 1 of Alice in Wonderland.

Usage:
    PYTHONPATH=. python scripts/test_ctc_align.py
"""

import re

import torch
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

device = "cpu"
audio_path = "audio/alice_in_wonderland/chapter_001.mp3"

text = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her "
    "sister was reading, but it had no pictures or conversations in it, "
    "and what is the use of a book, thought Alice without pictures or conversations?"
)

print("Loading model...")
alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=torch.float32,
)
print("Vocab:", sorted(alignment_tokenizer.get_vocab().keys()))

print("Loading audio...")
audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)

print("Generating emissions...")
emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=4)
print("Emissions shape after generate:", emissions.shape)

print("Preprocessing text...")
clean_text = re.sub(r"[^\w\s]", " ", text)
clean_text = re.sub(r"\s+", " ", clean_text).strip()
print(f"  Clean text: {clean_text[:80]}...")

tokens_starred, text_starred = preprocess_text(
    clean_text,
    romanize=True,
    language="eng",
)

print("Aligning...")
print("Text starred sample:", text_starred[:5])
print("Tokens starred sample:", tokens_starred[:5])


print("Emissions shape:", emissions.shape)
vocab = alignment_tokenizer.get_vocab()
print("Vocab size:", len(vocab))

for k, v in sorted(vocab.items(), key=lambda x: x[1]):
    print(f"  {v:3d}: {k}")

segments, scores, blank_token = get_alignments(
    emissions=emissions,
    tokens=tokens_starred,
    tokenizer=alignment_tokenizer,
)

spans = get_spans(tokens=tokens_starred, segments=segments, blank=blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)

print(f"\nGot {len(word_timestamps)} words:\n")
for i, word in enumerate(word_timestamps):
    print(f"  {i:3d}: {word['text']:20s} {word['start']:.3f}s - {word['end']:.3f}s")
