# Intertext Backend

Ingests novels into a PostgreSQL database (Supabase) with vector embeddings for similarity search. Splits text into semantically coherent segments, extracts literary metadata via OpenAI, stores embeddings for retrieval, and supports audio narration with word-level alignment.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

- `DATABASE_URL` — Supabase PostgreSQL connection string
- `DATABASE_TEST_URL` — test database connection string (optional)
- `OPENAI_API_KEY` — used for embeddings and metadata extraction
- `HF_TOKEN` — Hugging Face token (for downloading alignment models)
- `FRONTEND_URL` — frontend origin for CORS (defaults to `http://localhost:5173`)
- `AUDIO_BASE_URL` — base URL for serving audio files (defaults to `http://localhost:8000/audio/`)

3. Run database migrations:

```bash
alembic upgrade head
```

## Ingesting a Novel

Place a `.md` file in `books/`, then create `scripts/ingest_novel_inputs.json` (see `ingest_novel_inputs.example.json`):

```json
{
  "file_path": "books/jane_eyre.md"
}
```

Then run:

```bash
bash scripts/ingest_novel.sh
```

The script splits each chapter into segments using semantic chunking, extracts metadata (themes, characters, setting, mood) via `gpt-4o-mini`, computes embeddings with `text-embedding-3-large`, and stores everything in the database.

The process is resumable — if it fails partway through, re-running the script for the same title picks up from the last incomplete chapter.

## Deleting a Novel

```bash
bash scripts/delete_novel.sh "Jane Eyre"
```

Deletes the novel and all associated segments.

## Audio Narration

Audio narration uses chapter-level MP3 recordings aligned to text segments using CTC forced alignment (`ctc-forced-aligner`). The pipeline produces word-level timestamps stored in the `segment_audio` table.

### Audio assets

Audio files live in `audio/<novel_slug>/` with a `manifest.json` mapping chapters to MP3 files. See `audio/README.md` for the folder structure and how to add audio for a new novel. MP3 files are gitignored; manifests are tracked.

### Running alignment

```bash
bash scripts/align_audio.sh --novel-id <id>
```

Options:
- `--chapter <block_index>` — align a single chapter instead of all
- `--force` — re-align chapters that already have alignment data

The script loads the Wav2Vec2 model, performs forced alignment of chapter audio against segment text, and writes word-level timings (`char_start`, `char_end`, `start_ms`, `end_ms`) to the database. A post-alignment heuristic corrects words that may have been misaligned to preamble audio (e.g. LibriVox intros).

## Database Migrations

After changing `models.py`, generate and apply a migration:

```bash
alembic revision --autogenerate -m "description of change"
alembic upgrade head
```

Alembic reads `DATABASE_URL` from `.env` automatically.

## API

Start the server:

```bash
bash scripts/run_server.sh
```

Endpoints:

- `GET /api/novels` — list all novels
- `GET /api/novels/{novel_id}/chapters` — chapter listing with metadata
- `GET /api/novels/{novel_id}/segments` — all segments for a novel, ordered by position
- `GET /api/segments/random?count=5` — random segment previews
- `GET /api/segments/{segment_id}` — full segment with navigation, audio URL, and word timings
- `GET /api/segments/{segment_id}/similar?limit=3` — similar segments by embedding similarity
- `POST /api/segments/similar` — find segments similar to a given embedding vector
- `GET /audio/{file_path}` — serves audio files from the `audio/` directory
