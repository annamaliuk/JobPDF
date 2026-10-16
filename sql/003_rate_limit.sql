-- Rate-limit counters (JM-30). Idempotent: safe to run on every connect.
--
-- One row per (scope, subject, window): "key" + an API key's name, or "ip" + a keyed hash
-- of the client address (never the address itself). window_start is the window's first
-- second (unix time); hits is how many uploads were allowed in it. Rows are tiny and
-- PostgresCounterStore.purge() deletes the ones whose window ended long ago.

CREATE TABLE IF NOT EXISTS rate_limit_counters (
    scope          text    NOT NULL,
    subject        text    NOT NULL,
    window_seconds integer NOT NULL,
    window_start   bigint  NOT NULL,
    hits           integer NOT NULL,
    PRIMARY KEY (scope, subject, window_seconds, window_start)
);
