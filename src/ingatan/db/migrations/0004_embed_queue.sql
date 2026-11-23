-- Ingatan schema, version 4: facts that were saved before their vector existed.

-- With `[embeddings] defer = true`, `remember` saves a fact without calling the model and
-- leaves one row here. A background step of the server embeds the fact, stores the vector,
-- runs the meaning-based conflict check that `remember` could not run without the vector
-- (only when `check_conflicts` is 1: the caller may have asked for none), and deletes the
-- row. The row holds the fact's id and that one flag, never its text. `forget` deletes it
-- with the fact, and a row whose fact was closed in the meantime is dropped unprocessed.
CREATE TABLE embed_queue (
  fact_id TEXT PRIMARY KEY,
  check_conflicts INTEGER NOT NULL DEFAULT 1 CHECK (check_conflicts IN (0, 1)),
  created_at TEXT NOT NULL
);
