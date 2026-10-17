-- Vacancy store (JM-32). Idempotent: safe to run on every connect.
--
-- One row per posting, keyed on the source and the source's own posting ID, so a posting that
-- is fetched again updates its row instead of adding one. A row holds the vacancy as ranking
-- reads it (VacancyFeatures as JSON, the embedding beside it) and what is needed to decide
-- later whether it must be processed again:
--   content_hash       fingerprint of the posting's content (see jobpdf/ingestion/store.py)
--   *_version          what produced the embedding, the requirements (extraction_version,
--                      JM-48) and the skill IDs; a mismatch with the current versions marks
--                      the row stale
--   fetched_at         when the SOURCE data was fetched (not when it was written); an older
--                      fetch never overwrites a newer one
--   first_seen_at / last_seen_at   from fetch times too; a stale cache cannot make a posting
--                      look fresher than it is
-- The embedding is real[] (not pgvector): ranking compares vectors in Python, so no vector
-- search is needed here. No description text is kept, only what ranking uses.

CREATE TABLE IF NOT EXISTS vacancies (
    source                 text        NOT NULL CHECK (source <> ''),
    source_id              text        NOT NULL CHECK (source_id <> ''),
    content_hash           text        NOT NULL,
    features               jsonb       NOT NULL,
    embedding              real[],
    embedding_version      text,
    vacancy_text_version   text,
    taxonomy_version       text,
    index_version          text,
    matcher_prompt_version text,
    extraction_version     text,
    url                    text        NOT NULL,
    posted_at              timestamptz,
    fetched_at             timestamptz NOT NULL,
    first_seen_at          timestamptz NOT NULL,
    last_seen_at           timestamptz NOT NULL,
    updated_at             timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, source_id)
);
-- JM-48: tables created before the column existed get it here.
ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS extraction_version text;
CREATE INDEX IF NOT EXISTS vacancies_last_seen_idx ON vacancies (last_seen_at DESC);
