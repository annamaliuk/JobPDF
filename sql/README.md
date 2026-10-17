# Database migrations

The files in this folder are the database schema, applied in order by `scripts/migrate.py`
(`jobpdf/migrations.py`). `docker compose up` runs it for you, before the app and Airflow start.

```bash
uv run python scripts/migrate.py            # apply what is pending
uv run python scripts/migrate.py --status   # every migration and its state
uv run python scripts/migrate.py --check    # change nothing; exit 1 unless all are applied
docker compose run --rm migrate python scripts/migrate.py --status    # the same, in Docker
```

## Adding a change

1. Create `sql/<number>_<name>.sql`: the next number after the highest one (three or more
   digits), then an underscore and a lowercase name with letters, digits and underscores.
   Example: `sql/005_vacancy_remote_flag.sql`.
2. Write plain SQL. Prefer `IF NOT EXISTS` / `IF EXISTS` where it costs nothing: it makes the
   file safe to run against a database that already has the change.
3. Run `scripts/migrate.py` and the tests. Commit the file with the code that needs it.

Rules that the runner enforces:

- **Never edit a migration that has been applied anywhere** (merged to `main`, or run on a
  server). The run stops with `checksum_mismatch`. Change the schema with a new file. Line
  endings do not count as an edit.
- **Never delete or rename one.** The run stops with `missing_file`.
- **One migration is one transaction** with its record in `schema_migrations`: it applies
  completely or not at all. So do not use statements that refuse to run in a transaction
  (`CREATE INDEX CONCURRENTLY`, `VACUUM`).
- **Numbers may have gaps** and may arrive out of order: if two branches add `005` and `006` and
  `006` is deployed first, `005` is still applied on the next run.
- **Any other `.sql` file in this folder is an error** (`bad_name`), so a typo is not skipped.
  Two files with the same number are an error too (`duplicate_version`).
- Runs take turns (a Postgres advisory lock), so starting two at once is safe. One that waits
  more than 60 seconds stops with `lock_timeout`.

## The files so far

| Number | What |
|---|---|
| 001 | skill taxonomy and its vector index (needs the `vector` extension and a role allowed to create it) |
| 002 | match cache (JM-19; a checkout without it simply has a gap) |
| 003 | rate-limit counters |
| 004 | vacancies |

All of them can be applied to a database that already has the tables, which is how an existing
installation adopts the runner: the first run applies them again without changing anything and
records them.

## When a run fails

The message gives a code and a version, never SQL or a database error (they can contain data).
Fix the cause and run again; migrations that finished before the failure stay applied.

| Code | Meaning |
|---|---|
| `migration_failed 004` | the SQL in that file failed; run it by hand in `psql` to see why |
| `checksum_mismatch 003` | an applied file was edited: restore it (`git checkout`) and add a new migration |
| `missing_file 003` | the database has a migration this checkout does not: you are on an older version of the code |
| `bad_name`, `duplicate_version`, `no_directory` | the folder is wrong |
| `lock_timeout` | another run holds the lock; wait for it or look for a stuck session |
