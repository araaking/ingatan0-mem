-- Ingatan schema, version 5: which ingested notes the optional fact extraction has read.

-- `ingatan extract --documents` lets a model the user configured read ingested notes and
-- propose facts from them. `extracted_at` is set when a note was read, so it is not sent to
-- the model again. A file that changed is ingested as new rows, which start at NULL and
-- are read again. The column holds a timestamp and nothing of the note's text.
ALTER TABLE notes ADD COLUMN extracted_at TEXT;
