# Audio Assets

Chapter-level narration audio for use with the alignment pipeline. MP3 files are gitignored while manifests are tracked.

## Folder structure

```
audio/
  <novel_slug>/
    manifest.json       # chapter-to-file mapping (tracked in git)
    chapter_001.mp3     # audio files (gitignored)
    chapter_002.mp3
    ...
```

## Obtaining audio files

### Alice in Wonderland

Source: [LibriVox](https://librivox.org/alices-adventures-in-wonderland-by-lewis-carroll/)

Download the individual chapter MP3s and place them in `audio/alice_in_wonderland/` with filenames matching `manifest.json` entries (`chapter_001.mp3` through `chapter_012.mp3`).

## Adding a new novel

1. Create a folder under `audio/` using a slug (lowercase, underscores).
2. Add chapter MP3 files named `chapter_XXX.mp3` (zero-padded).
3. Add a `manifest.json` with `novel_id`, `novel_title`, `novel_slug`, and a `chapters` array mapping each `block_index` to its audio file.
