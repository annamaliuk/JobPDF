-- Skill taxonomy index (JM-17). Idempotent: safe to run on every build.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS taxonomy_alias (
    id             bigserial PRIMARY KEY,
    canonical_id   text NOT NULL,
    canonical_name text NOT NULL,
    alias          text NOT NULL,
    alias_norm     text NOT NULL,
    lang           text NOT NULL,
    alias_kind     text NOT NULL,
    exact_only     boolean NOT NULL,
    embedding      vector(384)            -- NULL for exact_only rows
);
CREATE INDEX IF NOT EXISTS taxonomy_alias_norm_idx ON taxonomy_alias (alias_norm);
CREATE INDEX IF NOT EXISTS taxonomy_alias_hnsw_idx
    ON taxonomy_alias USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS index_meta (
    key   text PRIMARY KEY,
    value jsonb NOT NULL
);
