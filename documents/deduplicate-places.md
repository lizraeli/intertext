# Deduplicating Places

During novel ingestion, the LLM may produce slightly different names for the same entity across chunks — for example, "red-room" and "the red-room" as separate places. This results in duplicate rows in entity tables like `novel_places` and duplicate associations in `novel_segments`.

The queries below help identify and merge these duplicate places without re-running the ingestion pipeline.

## Schema context

- `novel_places` has columns `id`, `novel_id`, `name` (unique on `(novel_id, name)`).
- `novel_segments` has a `place_id` FK pointing to `novel_places.id` (`ondelete="RESTRICT"`).
- Because of `RESTRICT`, you must reassign segments before deleting a place.

## Finding duplicates with heuristic normalization

This query strips leading articles ("the", "a", "an"), lowercases, and groups by the normalized name. It returns groups where more than one `novel_places` row maps to the same normalized form.

```sql
WITH normalized AS (
  SELECT
    id, novel_id, name,
    LOWER(TRIM(REGEXP_REPLACE(name, '^(the|a|an)\s+', '', 'i'))) AS norm
  FROM novel_places
),
dup_groups AS (
  SELECT
    novel_id, norm,
    (ARRAY_AGG(id ORDER BY LENGTH(name) DESC))[1] AS canonical_id,
    ARRAY_AGG(id) AS all_ids,
    ARRAY_AGG(name ORDER BY LENGTH(name) DESC) AS all_names
  FROM normalized
  GROUP BY novel_id, norm
  HAVING COUNT(*) > 1
)
SELECT * FROM dup_groups;
```

Review the output before proceeding. The longest name is chosen as canonical (first in `all_names`).

## Merging duplicates

Run these two statements in order. The UPDATE reassigns all segments from duplicates to the canonical place, then the DELETE removes the now-orphaned rows.

### Step 1 — Reassign segments

```sql
WITH normalized AS (
  SELECT
    id, novel_id, name,
    LOWER(TRIM(REGEXP_REPLACE(name, '^(the|a|an)\s+', '', 'i'))) AS norm
  FROM novel_places
),
dup_groups AS (
  SELECT
    novel_id, norm,
    (ARRAY_AGG(id ORDER BY LENGTH(name) DESC))[1] AS canonical_id
  FROM normalized
  GROUP BY novel_id, norm
  HAVING COUNT(*) > 1
),
duplicates AS (
  SELECT n.id AS duplicate_id, dg.canonical_id
  FROM normalized n
  JOIN dup_groups dg ON n.novel_id = dg.novel_id AND n.norm = dg.norm
  WHERE n.id != dg.canonical_id
)
UPDATE novel_segments
SET place_id = d.canonical_id
FROM duplicates d
WHERE novel_segments.place_id = d.duplicate_id;
```

### Step 2 — Delete orphaned duplicates

```sql
WITH normalized AS (
  SELECT
    id, novel_id, name,
    LOWER(TRIM(REGEXP_REPLACE(name, '^(the|a|an)\s+', '', 'i'))) AS norm
  FROM novel_places
),
dup_groups AS (
  SELECT
    novel_id, norm,
    (ARRAY_AGG(id ORDER BY LENGTH(name) DESC))[1] AS canonical_id
  FROM normalized
  GROUP BY novel_id, norm
  HAVING COUNT(*) > 1
),
duplicates AS (
  SELECT n.id AS duplicate_id, dg.canonical_id
  FROM normalized n
  JOIN dup_groups dg ON n.novel_id = dg.novel_id AND n.norm = dg.norm
  WHERE n.id != dg.canonical_id
)
DELETE FROM novel_places
WHERE id IN (SELECT duplicate_id FROM duplicates);
```

## Merging a specific pair manually

If you've identified two places that should be unified (e.g., via the frontend or by browsing the data), you can merge them directly by ID.

### Look up IDs by name

```sql
SELECT id, name FROM novel_places
WHERE novel_id = :your_novel_id
  AND name IN ('the red-room', 'red-room');
```

### Merge

Replace `:canonical_id` with the ID to keep and `:duplicate_id` with the one to remove.

```sql
UPDATE novel_segments SET place_id = :canonical_id WHERE place_id = :duplicate_id;
DELETE FROM novel_places WHERE id = :duplicate_id;
```
