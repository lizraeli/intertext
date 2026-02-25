# Intertext Backend

Ingests novels into a PostgreSQL database (Supabase) with vector embeddings for similarity search. Splits text into semantically coherent segments, extracts literary metadata via OpenAI, and stores embeddings for retrieval.

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
- `OPENAI_API_KEY` — used for embeddings and metadata extraction
- `HF_TOKEN` — Hugging Face token (for model downloads)

3. Run database migrations:

```bash
alembic upgrade head
```

## Ingesting a Novel

Place a `.md` file in `books/`, then edit `scripts/ingest_novel_inputs.json` with the book details:

```json
{
  "title": "Jane Eyre",
  "author": "Charlotte Brontë",
  "year": 1847,
  "file_path": "books/jane_eyre.md"
}
```

Then run:

```bash
bash scripts/injest_novel.sh
```

The script splits each chapter into segments using semantic chunking, extracts metadata (themes, characters, setting, mood) via `gpt-4o-mini`, computes embeddings with `text-embedding-3-large`, and stores everything in the database.

The process is resumable — if it fails partway through, re-running the script for the same title picks up from the last incomplete chapter.

## Deleting a Novel

```bash
bash scripts/delete_novel.sh "Jane Eyre"
```

Deletes the novel and all associated segments.

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
uvicorn main:app --reload
```

Endpoints:

- `GET /api/novels/{novel_id}/segments` — returns all segments for a novel, ordered by position
- `POST /api/segments/similar` — finds segments similar to a given embedding vector, with optional theme filtering
