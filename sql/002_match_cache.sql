-- Skill match cache (JM-19). Idempotent: safe to run on every connect.
--
-- cache_key is match_cache.cache_key(): "<index version>|<prompt version>|<normalized skill>".
-- The versions are part of the key, so a rebuilt index or a new prompt simply stops
-- finding old rows; PostgresSkillMatchCache.prune_other_versions() deletes them.

CREATE TABLE IF NOT EXISTS skill_match_cache (
    cache_key  text PRIMARY KEY,
    match      jsonb NOT NULL,          -- SkillMatch.model_dump_json()
    created_at timestamptz NOT NULL DEFAULT now()
);
