-- Ingatan schema, version 2: semantic embeddings, cognitive categories, aliases, and relation edges.

ALTER TABLE facts ADD COLUMN category TEXT NOT NULL DEFAULT 'fact'
  CHECK (category IN ('fact', 'rule', 'procedure', 'preference'));

ALTER TABLE facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0
  CHECK (pinned IN (0, 1));

CREATE INDEX facts_category ON facts (category);
CREATE INDEX facts_pinned ON facts (pinned);

-- L2-normalized float32 vectors stored as raw little-endian BLOBs inside the same SQLite file,
-- preserving single-file isolation and byte-level `forget` via PRAGMA secure_delete = ON.
CREATE TABLE fact_embeddings (
  fact_id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  dims INTEGER NOT NULL,
  vec BLOB NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE note_embeddings (
  note_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  dims INTEGER NOT NULL,
  vec BLOB NOT NULL,
  updated_at TEXT NOT NULL
);

-- Project-scoped terminology glossary (e.g. "LP" -> "landing page").
CREATE TABLE aliases (
  term TEXT NOT NULL,
  canonical TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (term, canonical)
);

-- Lightweight relation graph between facts (stored in business.db).
CREATE TABLE edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_id TEXT NOT NULL,
  dst_id TEXT NOT NULL,
  relation TEXT NOT NULL CHECK (relation IN ('related_to', 'part_of')),
  created_at TEXT NOT NULL,
  UNIQUE (src_id, dst_id, relation)
);

CREATE INDEX edges_src ON edges (src_id);
CREATE INDEX edges_dst ON edges (dst_id);
