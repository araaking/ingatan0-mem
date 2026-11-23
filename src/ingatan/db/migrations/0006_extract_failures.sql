-- Ingatan schema, version 6: how often the optional fact extraction failed on a text.

-- `ingatan extract` reads pending episodes and unread notes oldest first. A text the model
-- cannot be read for (an answer that is not JSON, a request the endpoint refuses) used to be
-- tried again on every run, and a few of them at the head kept everything newer from being
-- read. `extract_failures` counts those failed readings; after three the text is left out
-- until `ingatan extract --retry-failed`. The column holds a count and nothing of the text.
ALTER TABLE episodes ADD COLUMN extract_failures INTEGER NOT NULL DEFAULT 0;
ALTER TABLE notes ADD COLUMN extract_failures INTEGER NOT NULL DEFAULT 0;
