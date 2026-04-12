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

Each novel's `manifest.json` includes `source_url` and `reader` fields indicating where to download the chapter MP3s. Place the downloaded files in the novel's folder with filenames matching the `file` entries in the manifest.

## Adding a new novel

1. Create a folder under `audio/` using a slug (lowercase, underscores).
2. Add chapter MP3 files named `chapter_XXX.mp3` (zero-padded).
3. Add a `manifest.json` with `novel_id`, `novel_title`, `novel_slug`, `source_url`, `reader`, and a `chapters` array mapping each `block_index` to its audio file.
