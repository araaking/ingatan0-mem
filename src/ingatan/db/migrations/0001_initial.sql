-- Ingatan schema, version 1. Used by every database file (business and global).

-- `seq` is an explicit INTEGER PRIMARY KEY, so it is the rowid and stays stable under
-- VACUUM (the implicit rowid of a table with a TEXT key may be renumbered). The FTS
-- indexes below use it as their rowid.
CREATE TABLE facts (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  text TEXT NOT NULL,
  entities TEXT NOT NULL,
  keywords TEXT NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('global', 'business', 'agent')),
  agent TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('stable', 'volatile')),
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  superseded_by TEXT,
  review_after TEXT,
  review_interval_days INTEGER,
  pending_proposal TEXT,
  source TEXT NOT NULL,
  use_count INTEGER NOT NULL DEFAULT 0,
  last_used TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX facts_tier_agent ON facts (tier, agent);
CREATE INDEX facts_valid_to ON facts (valid_to);
CREATE INDEX facts_superseded_by ON facts (superseded_by);
CREATE INDEX facts_review_after ON facts (review_after);
CREATE INDEX facts_pending_proposal ON facts (pending_proposal);

-- Full text indexes. They hold their own copy of the text (no external content),
-- kept in step with `facts` by the triggers below. The rowid is facts.rowid.
-- Entities and keywords are JSON arrays; the indexes get them as plain words.
CREATE VIRTUAL TABLE facts_fts USING fts5(
  text, entities, keywords,
  tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO facts_fts (facts_fts, rank) VALUES ('secure-delete', 1);
CREATE VIRTUAL TABLE facts_tri USING fts5(
  text, entities, keywords,
  tokenize = 'trigram'
);
INSERT INTO facts_tri (facts_tri, rank) VALUES ('secure-delete', 1);

-- FTS5 secure-delete (SQLite 3.42+): deleted terms are removed from the index pages
-- instead of being tombstoned, which `forget` relies on.

CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts (rowid, text, entities, keywords) VALUES (
    new.rowid, new.text,
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.entities)), ''),
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.keywords)), ''));
  INSERT INTO facts_tri (rowid, text, entities, keywords) VALUES (
    new.rowid, new.text,
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.entities)), ''),
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.keywords)), ''));
END;

CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
  DELETE FROM facts_fts WHERE rowid = old.rowid;
  DELETE FROM facts_tri WHERE rowid = old.rowid;
END;

CREATE TRIGGER facts_au AFTER UPDATE OF text, entities, keywords ON facts BEGIN
  DELETE FROM facts_fts WHERE rowid = old.rowid;
  DELETE FROM facts_tri WHERE rowid = old.rowid;
  INSERT INTO facts_fts (rowid, text, entities, keywords) VALUES (
    new.rowid, new.text,
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.entities)), ''),
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.keywords)), ''));
  INSERT INTO facts_tri (rowid, text, entities, keywords) VALUES (
    new.rowid, new.text,
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.entities)), ''),
    COALESCE((SELECT group_concat(value, ' ') FROM json_each(new.keywords)), ''));
END;

CREATE TABLE conflicts (
  id INTEGER PRIMARY KEY,
  fact_a TEXT NOT NULL,
  fact_b TEXT NOT NULL,
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  resolution TEXT
);

CREATE INDEX conflicts_fact_a ON conflicts (fact_a);
CREATE INDEX conflicts_fact_b ON conflicts (fact_b);
CREATE INDEX conflicts_unresolved ON conflicts (resolved_at);

CREATE TABLE log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  agent TEXT NOT NULL,
  op TEXT NOT NULL,
  fact_id TEXT,
  tier TEXT,
  owner TEXT,
  change TEXT
);

CREATE INDEX log_fact_id ON log (fact_id);
CREATE INDEX log_ts ON log (ts);

-- FR-36: the log is append-only whatever the Python code does. The one allowed
-- change to an existing row is clearing `change` (the scrub done by forget).
CREATE TRIGGER log_no_delete BEFORE DELETE ON log BEGIN
  SELECT RAISE(ABORT, 'log entries cannot be deleted');
END;

CREATE TRIGGER log_no_update BEFORE UPDATE ON log
WHEN NOT (
  new.change IS NULL
  AND new.id IS old.id
  AND new.ts IS old.ts
  AND new.agent IS old.agent
  AND new.op IS old.op
  AND new.fact_id IS old.fact_id
  AND new.tier IS old.tier
  AND new.owner IS old.owner
)
BEGIN
  SELECT RAISE(ABORT, 'log entries are append-only; only the change can be cleared');
END;

CREATE TABLE proposals (
  id TEXT PRIMARY KEY,
  tier TEXT NOT NULL CHECK (tier IN ('business', 'global')),
  payload TEXT NOT NULL,
  similar TEXT NOT NULL,
  supersedes TEXT,
  agent TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
  decided_at TEXT,
  fact_id TEXT
);

CREATE INDEX proposals_status ON proposals (status);

CREATE TABLE episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent TEXT NOT NULL,
  session TEXT,
  user_text TEXT NOT NULL,
  assistant_text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'processed')),
  offered INTEGER NOT NULL DEFAULT 0,
  processed_at TEXT
);

CREATE INDEX episodes_agent_status ON episodes (agent, status);

CREATE TABLE notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  heading TEXT NOT NULL,
  text TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX notes_path ON notes (path);

CREATE VIRTUAL TABLE notes_fts USING fts5(
  heading, text,
  tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO notes_fts (notes_fts, rank) VALUES ('secure-delete', 1);
CREATE VIRTUAL TABLE notes_tri USING fts5(
  heading, text,
  tokenize = 'trigram'
);
INSERT INTO notes_tri (notes_tri, rank) VALUES ('secure-delete', 1);

CREATE TRIGGER notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts (rowid, heading, text) VALUES (new.id, new.heading, new.text);
  INSERT INTO notes_tri (rowid, heading, text) VALUES (new.id, new.heading, new.text);
END;

CREATE TRIGGER notes_ad AFTER DELETE ON notes BEGIN
  DELETE FROM notes_fts WHERE rowid = old.id;
  DELETE FROM notes_tri WHERE rowid = old.id;
END;

CREATE TRIGGER notes_au AFTER UPDATE OF heading, text ON notes BEGIN
  DELETE FROM notes_fts WHERE rowid = old.id;
  DELETE FROM notes_tri WHERE rowid = old.id;
  INSERT INTO notes_fts (rowid, heading, text) VALUES (new.id, new.heading, new.text);
  INSERT INTO notes_tri (rowid, heading, text) VALUES (new.id, new.heading, new.text);
END;
