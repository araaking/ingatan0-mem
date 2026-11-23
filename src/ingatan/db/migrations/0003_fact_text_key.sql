-- Ingatan schema, version 3: a lookup key for "is this exact text already stored?".

-- `text_key` is a hash of the fact's text with case, spacing and trailing punctuation
-- removed (`ingatan.db.fact_text_key`). `remember` uses it to find a repeated fact with one
-- index lookup instead of reading every fact. Rows written before this version have NULL
-- until `db.connect` fills them in; until then such a row is simply not found as a repeat.
-- The key never leaves the row: it is not copied into the log or into a proposal, so
-- deleting the row (which `forget` does, with secure_delete on) removes it too.
ALTER TABLE facts ADD COLUMN text_key TEXT;

CREATE INDEX facts_text_key ON facts (text_key);
