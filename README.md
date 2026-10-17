# JobPDF

Upload a CV, get remote vacancies ranked by how well they fit. The CV is read by an LLM, its
skills are matched against the ESCO taxonomy, and each stored vacancy gets a fit score and a
band (strong, partial, weak, unscored). Nothing about a CV is stored.

One Python package, `jobpdf`, runs as two processes that share one Postgres database:

| Process | What it does | Started by |
|---|---|---|
| **app** (FastAPI) | the web page (`GET /`, `POST /report`), `POST /v1/analyze`, `GET /health`, `GET /metrics` | `uvicorn` / docker compose |
| **ingestion** (Airflow DAGs) | fetch postings from the job sources, normalize, embed, store them | Airflow, or by hand with `scripts/` |

```
 job sources ──► Airflow ──► Postgres ◄── app ◄── CV upload (web page, or API key)
 (Remotive, The Muse,        vacancies,         parse → extract → normalize → rank
  Arbeitnow, Adzuna,         skill index,
  Jooble)                    rate limits
```

## Contents

1. [Try it on your machine](#1-try-it-on-your-machine-no-docker-for-the-app)
2. [Run everything with docker compose](#2-run-everything-with-docker-compose)
3. [Deploy to a server](#3-deploy-to-a-server)
4. [Using the API](#4-using-the-api)
5. [Operating it](#5-operating-it)
6. [Settings](#6-settings)
7. [Continuous integration](#7-continuous-integration)
8. [Monitoring and alerts](#8-monitoring-and-alerts)

---

## 1. Try it on your machine (no Docker for the app)

You need Python 3.10+, [uv](https://docs.astral.sh/uv/) and Docker (for Postgres only).

```bash
cp .env.example .env            # then fill it in, see "Settings"
uv sync                         # installs the app and the dev tools
docker compose up -d db         # Postgres 16 with pgvector on 127.0.0.1:5432
uv run python scripts/migrate.py   # creates the tables (what it does: sql/README.md)
```

The skill index needs the ESCO taxonomy once. Download the ESCO skills export (CSV, English and
Ukrainian) from the ESCO portal, then:

```bash
uv run python scripts/build_taxonomy.py --zip path/to/esco.zip   # writes data/taxonomy/
uv run python scripts/build_index.py                              # embeds it into Postgres (minutes)
```

Put some vacancies in the database (see [Filling it with vacancies](#filling-it-with-vacancies)), make
an API key, and start the app:

```bash
uv run python scripts/make_api_key.py me            # prints the key once + the line for .env
# paste the "me:…" line into JOBPDF_API_KEYS in .env, set JOBPDF_RATE_LIMIT_SECRET and ANTHROPIC_API_KEY

uv run uvicorn jobpdf.api.app:create_app --factory --no-access-log
curl localhost:8000/health
```

The first start downloads the embedding model (about 470 MB) and takes a minute.

Checks before pushing: `uv run ruff check .` and `uv run pytest` (what CI runs, see
[section 7](#7-continuous-integration)). Tests need no network and no API key; the tests marked
`db` and `model` skip themselves when Postgres or the model are missing.

---

## 2. Run everything with docker compose

This starts Postgres, the app and an HTTPS proxy (Caddy). Airflow is optional (`--profile airflow`).
Docker Compose v2 is required.

### First-time setup

**1. Settings.** `cp .env.example .env` and fill in:

| Variable | What to put |
|---|---|
| `POSTGRES_PASSWORD` | a long random password (URL-safe characters only, it is used in a URL) |
| `ANTHROPIC_API_KEY` | your Anthropic key |
| `JOBPDF_RATE_LIMIT_SECRET` | 16+ random characters: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `APP_UID` | `id -u` (so the containers can write to `./data`) |
| `DOMAIN` | `localhost` to try it, your real name on a server |

**2. Build the images** (a few minutes the first time; the embedding model is baked in):

```bash
docker compose build
```

**3. Start the database, create the tables and build the skill index** (once; repeat only when the
taxonomy changes):

```bash
docker compose up -d db
docker compose run --rm migrate                    # creates the tables, see sql/README.md
cp /path/to/esco.zip data/esco.zip
docker compose run --rm tools python scripts/build_taxonomy.py --zip data/esco.zip
docker compose run --rm tools python scripts/build_index.py
```

**4. Make an API key** and give the server its hash:

```bash
docker compose run --rm tools python scripts/make_api_key.py partner
```

The script prints the key (give it to the caller; it is not stored anywhere) and a
`partner:<sha256>` line. Put that line in `JOBPDF_API_KEYS` in `.env`. Several keys: separate the
entries with commas.

**5. Fill it with vacancies** (see below), then **start it**:

```bash
docker compose up -d
docker compose ps                                  # app should become "healthy" within ~2 minutes
curl -k https://localhost/health                   # -k only because "localhost" uses a local certificate
```

### Filling it with vacancies

Vacancies come from the job sources, are normalized and embedded once, and are stored. A CV
request only reads them. By hand (these use the same cache and request guards as Airflow, so
they cannot overspend a source's limit):

```bash
docker compose run --rm tools python scripts/fetch_remotive.py       # Remotive: at most 4 requests a day
docker compose run --rm tools python scripts/fetch_sources.py        # The Muse, Arbeitnow, Adzuna, Jooble (those with keys)
docker compose run --rm tools python scripts/ingest_vacancies.py     # cached postings → vacancies table
```

Every 8 hours, automatically, with Airflow:

```bash
# in .env: AIRFLOW_ADMIN_PASSWORD and AIRFLOW_SECRET_KEY
docker compose --profile airflow up -d --build
ssh -L 8080:localhost:8080 your-server     # the UI listens on 127.0.0.1 only; then open http://localhost:8080
```

The three DAGs are unpaused when created: `remotive_fetch` (minute 0), `job_sources_fetch`
(minute 20), `vacancy_ingest` (minute 50), every 8 hours. For the very first data, trigger them
in that order from the UI instead of waiting.

Without Airflow, the same three scripts from the host's crontab do the job:

```cron
0  */8 * * *  cd /srv/jobpdf && docker compose run --rm -T tools python scripts/fetch_remotive.py
20 */8 * * *  cd /srv/jobpdf && docker compose run --rm -T tools python scripts/fetch_sources.py
50 */8 * * *  cd /srv/jobpdf && docker compose run --rm -T tools python scripts/ingest_vacancies.py
```

### What runs where

| Service | Port | Notes |
|---|---|---|
| `proxy` (Caddy) | 80, 443 | the only public entry; HTTPS certificate is automatic; 6 MB body cap |
| `app` | 8000 (internal) | read-only file system, no Linux capabilities, uploads in memory (`tmpfs`) |
| `db` | 127.0.0.1:5432 | never exposed to the network |
| `migrate` | none | runs once on every `up`, applies new `sql/*.sql` files and exits; `app` and `airflow-init` wait for it |
| `airflow-*` | 127.0.0.1:8080 | only with `--profile airflow` |
| `tools` | none | `docker compose run --rm tools python scripts/<name>.py` |

`./data` is shared by all of them: the taxonomy files, the fetch cache (`data/cache/`) and the
ESCO export. It is not in git.

---

## 3. Deploy to a server

A single Linux machine with Docker is enough. Sizing is an estimate, not a measurement: plan for
**4 GB of RAM** for app + Postgres, and **8 GB** if Airflow runs there too (each of the app and
the ingestion task holds the embedding model and torch in memory). Measure with `docker stats`
after the first real CV.

1. **Machine and DNS.** Install Docker (with the Compose plugin). Point an `A` record for your
   name at the machine. Open ports 80 and 443 only; Postgres and Airflow stay on loopback.
2. **Code and settings.** `git clone` the repository to e.g. `/srv/jobpdf`, then follow
   [first-time setup](#first-time-setup) with `DOMAIN=jobs.example.com`. Caddy requests the
   certificate on first start; it needs ports 80 and 443 reachable from the internet.
3. **Start.** `docker compose up -d` (add `--profile airflow` for the scheduler). Every service
   has `restart: unless-stopped`, so they come back after a reboot once Docker starts.
4. **Check.** `curl https://jobs.example.com/health` → `"status": "ok"`.
5. **Monitor.** Point an uptime check at `https://<domain>/health`. It answers 200 for `ok` and
   `degraded`, 503 for `down`; the JSON says which check is not fine and why (a stable code).

**Updating.**

```bash
git pull
docker compose up -d --build            # rebuilds changed images, applies new migrations, restarts
```

Schema changes are migrations in `sql/` and are applied by the `migrate` service on that `up`
(`docker compose logs migrate` shows what it did). If one fails, `up` reports it and the new app
starts only after you fix it and run `up` again. See [sql/README.md](sql/README.md).

If the taxonomy or the embedding model changed, rebuild the index
(`docker compose run --rm tools python scripts/build_index.py`); the health report shows the
index and vacancies as stale until `vacancy_ingest` has re-embedded them.

**Backups.** The only state that matters is the Postgres volume (vacancies, the skill index,
rate-limit counters, Airflow history). Nothing about any CV is stored anywhere.

```bash
docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' | gzip > jobpdf-$(date +%F).sql.gz
```

**Secrets.** They live in `.env` on the server only (`chmod 600 .env`). Rotating an API key:
make a new one, replace its line in `.env`, `docker compose up -d app`. Logs and health
output show a key only as "set" or "missing".

**Logs.** `docker compose logs -f app`. They hold counts, codes and timings, never CV text,
keys or client addresses (uvicorn's access log is off for that reason).

---

## 4. Using the API

```bash
curl https://jobs.example.com/v1/analyze \
  -H "X-API-Key: $JOBPDF_KEY" \
  -F "file=@my-cv.pdf"
```

- One multipart field named `file`: a PDF or DOCX, at most 5 MB.
- The key goes in `X-API-Key` (or `Authorization: Bearer <key>`).
- Optional query parameters `page` (default 1) and `page_size` (default: everything, at most
  200). Nothing is stored between requests, so asking for another page runs the whole analysis
  again and counts against the rate limit.
- The answer is JSON (`AnalysisResponse`, schema in `/docs`): the profile, ranked matches with
  bands, score components and matched/missing skills, warnings (`code` + `detail`) and the
  versions that produced the result. It is sent with `Cache-Control: no-store`.

**Limits.** Per key: 6 uploads a minute and 100 a day. Not more than 4 CVs are analysed at once
on one server.

**Errors** always look like `{"error": {"code": "...", "stage": "...", "detail": "..."}}`:

| Status | `code` | Meaning |
|---|---|---|
| 400 | `invalid_request` | not a multipart form, or no `file` field |
| 401 | `missing_key`, `invalid_key` | |
| 413 | `too_large` | over 5 MB |
| 415 | `unsupported_type` | not a PDF or DOCX |
| 422 | `encrypted`, `corrupt`, `too_many_pages`, `no_text` | the file cannot be read |
| 429 | `rate_limited` | wait `Retry-After` seconds |
| 502 | `llm_error`, `extraction_invalid` | the language model failed; retry later |
| 503 | `llm_not_configured` | the server has no LLM key |
| 503 | `limiter_unavailable` | the limits cannot be checked (database down); uploads are refused on purpose |
| 503 | `busy` | all analysis slots are taken; retry after `Retry-After` |
| 500 | `internal_error` | a bug; it is counted in `/metrics` as `unexpected` |

Other endpoints:

- `GET /health`: public. `status` is `ok`, `degraded` or `down` (the worst of the checks).
  `down` means the database is unreachable or the LLM key is missing.
- `GET /metrics`: needs an API key. Counts of analyses, failures by `stage:code`, warnings,
  per-stage timings (p50, p95, max), token totals and refusals. In memory, reset on restart.

### The web page

Open the site's address in a browser (`https://<DOMAIN>/`, or `https://localhost/` locally).
Choose a PDF or DOCX of at most 5 MB and press the button; the report appears on the next
page: the figures from the CV, the best matches with "You have 7 of 9 required skills" and
what is missing, the skills that were found and how abbreviations were read, every match in a
table, notes on data quality and how the fit is scored. Close the page and it is gone.

- **One request, nothing stored.** The page is made by the server (`jobpdf/web/`, Jinja2
  templates, one stylesheet and one small script, no build step). It runs the same analysis as
  `/v1/analyze`: the same size cap, the same limit of 4 analyses at once, the same metrics. The
  CV text goes to the language model service (Anthropic), and the page says so before upload.
- **No key in the browser.** A key in a page's script is not a secret, so the page is limited
  per visitor address instead: 5 uploads an hour and 20 a day (the address is hashed with
  `JOBPDF_RATE_LIMIT_SECRET`, never stored). When the limit is reached the page says how long
  to wait. Several people behind one office address share it.
- **The address must reach the app.** Compose sets `FORWARDED_ALLOW_IPS` so that the app reads
  the visitor's address from the proxy's `X-Forwarded-For`. This is safe only while port 8000
  is not published. If you put another proxy or a load balancer in front of Caddy, or publish
  the app's port, review it; with a wrong value everybody counts as one address.
- **Pages never leave a trace.** They are sent with `Cache-Control: no-store`, no referrer, a
  Content-Security-Policy that forbids inline scripts and anything from other sites, and
  `X-Frame-Options: DENY`. Text from a CV or a vacancy is escaped, and a vacancy's link is
  shown only if it is an http(s) address.
- The page is in English. It does not show a progress bar (the browser waits for the report),
  and it is not part of the JSON API: `/docs` does not list it.

---

## 5. Operating it

| Symptom | Look at | Likely cause and fix |
|---|---|---|
| `docker compose up` stops with `migrate exited with code 1` | `docker compose logs migrate` | a migration failed or an applied file was edited; the message gives a code and a version, the table in `sql/README.md` says what to do |
| no alerts arrive although something is wrong | `docker compose logs monitor` | the monitor is not running (`--profile monitor`), `ALERT_WEBHOOK_URL` is empty (it then only prints), or the log says `webhook failed` |
| `app` never becomes healthy | `docker compose logs app` | `JOBPDF_RATE_LIMIT_SECRET` missing (the app refuses to start without it), or `db` not ready |
| `/health` → `database` `unreachable` | `docker compose ps db` | database is down or `POSTGRES_PASSWORD` differs from the volume's |
| `normalization` `index_unavailable` | | skill index not built: run `build_taxonomy.py` and `build_index.py` |
| `normalization` `llm_tier_disabled` | | `ANTHROPIC_API_KEY` missing: matching still works for exact and vector hits only |
| `llm` `key_missing` (status `down`) | `.env` | set `ANTHROPIC_API_KEY`, then `docker compose up -d app` |
| `vacancies` `no_vacancies` / `table_missing` | | run the ingestion once (see above) |
| `vacancies` `stale_vacancies`, `fetch` `stale_fetch` / `no_cache` | Airflow UI, `data/cache/` | the DAGs stopped or a source key is wrong; run the scripts by hand to see the error code |
| `fetch` `requests_24h` for Remotive near 4 | | Remotive allows 4 requests a day; the guard stops further requests by itself |
| matches are empty with warning `no_vacancies` | | the app cannot read vacancies; it keeps the last good copy and retries every 30 s |
| uploads answer 503 `limiter_unavailable` | `/health` | Postgres is down; uploads resume when it is back |
| the web page says "You have uploaded a lot of CVs" to everyone | `docker compose exec app env \| grep FORWARDED` | the app sees only the proxy's address: `FORWARDED_ALLOW_IPS` is missing or the app is not behind Caddy |

The app reloads vacancies from the database every 5 minutes, so new postings appear without a
restart.

---

## 6. Settings

All in `.env` (template: `.env.example`). Keys are never printed or logged.

| Variable | Used by | Purpose |
|---|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | db | the database; `POSTGRES_PORT` is the loopback port on the host |
| `DATABASE_URL` | scripts on the host | inside compose it is built from the values above |
| `ANTHROPIC_API_KEY` | app, ingestion | CV reading, skill matching, requirement extraction |
| `JOBPDF_API_KEYS` | app | `name:sha256` entries, comma-separated (or `JOBPDF_API_KEYS_FILE`) |
| `JOBPDF_RATE_LIMIT_SECRET` | app | hashes client addresses for the per-address limit (the web page); 16+ characters |
| `FORWARDED_ALLOW_IPS` | app | which proxies uvicorn trusts for the visitor's address; set in `docker-compose.yml` (`*`, because the port is not published), not in `.env` |
| `JOBPDF_SCORER` | app | `linear` (default) or `learned` |
| `THE_MUSE_API_KEY`, `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`, `JOOBLE_API_KEY` (+ optional query settings) | ingestion | a source without its key is skipped |
| `APP_UID` | compose | the user the containers run as |
| `DOMAIN` | proxy | public host name for the HTTPS certificate |
| `AIRFLOW_ADMIN_PASSWORD`, `AIRFLOW_SECRET_KEY` | airflow | the UI login and its session secret |
| `JOBPDF_MONITOR_API_KEY`, `ALERT_WEBHOOK_URL`, `ALERT_LABEL`, `MONITOR_IGNORE` | monitor | a key for `/metrics`, where alerts are posted, the name in messages, alert codes to ignore (section 8) |

---

## 7. Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to `main`. It needs no
secrets (tests use fakes, never the network or an API key), so pull requests from forks run the
same checks.

| Job | What it checks |
|---|---|
| **ruff + pytest** (Python 3.10 and 3.12) | `uv sync --locked`, `ruff check .`, then the tests against a real Postgres 16 with pgvector. The `db` tests run separately and the job fails if fewer than 50 of them actually ran, because they skip quietly when the database is missing. |
| **Airflow DAG tests** | installs Airflow 2.10.5 (the version in the Dockerfile) and runs `tests/ingestion/test_dag.py`, which skips without Airflow |
| **Docker images** | `docker compose config` for every profile; builds the `app` and `airflow` images; starts the app image and loads the embedding model offline; loads the three DAGs inside the Airflow image |

**Run the same checks yourself:**

```bash
uv sync --locked && uv run ruff check .
docker compose up -d db                                  # needs DATABASE_URL in .env
uv run pytest -m db -q --junitxml=junit-db.xml && uv run python scripts/check_db_tests_ran.py junit-db.xml
uv run pytest -m "not db" -q
```

**When it is red.** `uv sync --locked` failing means `uv.lock` does not match `pyproject.toml`:
run `uv lock` and commit the result. A failed "Database tests really ran" step means the
database service did not come up; the log of the "Database tests" step lists the skip reasons.

**Make it count.** In the repository settings (Branches → branch protection rule for `main`),
require these status checks before merging: `ruff + pytest (Python 3.10)`,
`ruff + pytest (Python 3.12)`, `Airflow DAG tests` and `Docker images`. The Docker job takes
several minutes on a cold cache (it downloads the embedding model); later runs reuse the cache.

---

## 8. Monitoring and alerts

`/health` and `/metrics` (section 5) tell you what is wrong when you look. The **monitor** looks
for you: every minute it asks the service, applies fixed rules, and posts to a chat webhook when
something starts, is still wrong after 6 hours, or is fixed. It is `scripts/uptime_check.py`, plain
Python with no extra service or account.

**Set it up** (about five minutes):

1. Make a key that may read `/metrics` and give the server its hash, like any caller (section 2, step 4):
   `docker compose run --rm tools python scripts/make_api_key.py monitor`. Put the **key** in `.env` as
   `JOBPDF_MONITOR_API_KEY` and the `monitor:<sha256>` line into `JOBPDF_API_KEYS`. Without the key only
   `/health` is checked.
2. Create an incoming webhook in Slack, Mattermost or Discord and put its https address in
   `ALERT_WEBHOOK_URL` (it contains a token: `.env` only). Without it the monitor only prints.
3. `docker compose up -d app` (to load the new key hash), then `docker compose --profile monitor up -d monitor`.
   `docker compose logs monitor` shows `monitor: metrics key set, webhook set` and, later, every change.

Try it: `docker compose stop app`; after about two minutes `CRITICAL unreachable` arrives. `docker
compose start app` brings `RESOLVED unreachable` about two minutes after it answers again.

**What is checked.** Codes are stable; a message is a fixed sentence with the code, never data from a CV.

| Alert | Severity | Means | What to do |
|---|---|---|---|
| `unreachable` | critical | `/health` does not answer (or the proxy says 502/503/504 without a report) | `docker compose ps`, `docker compose logs app` |
| `bad_response` | critical | an answer that is not a health report: wrong address, or something else on that port | check `MONITOR_URL` / `--url` |
| `check.<name>.<code>` | critical if the check is `down`, else warning | one per health check that is not ok, e.g. `check.database.database_down`, `check.vacancies.stale_vacancies`, `check.fetch.stale_fetch`, `check.llm.key_missing` | the table in section 5 |
| `health_down` | critical | the report says `down` but no single check does | `curl /health`, the logs |
| `service_errors` | warning at 20%, critical at 50% | that share of analyses failed **on our side**: LLM errors, invalid extractions, unexpected exceptions | `/metrics` `errors`, the LLM key and quota, `docker compose logs app` |
| `slow_analyses` | warning | the 95th percentile of a whole analysis is over 60 seconds | `/metrics` `stages` shows which stage |
| `token_spike` | warning | successful analyses average over 40 000 LLM tokens | `/metrics` `tokens`; a changed prompt or a stuck loop costs money |
| `limiter_failing` | warning | the rate limiter cannot reach its database; uploads are refused | the database |
| `metrics_unavailable` | warning | the key is rejected or `/metrics` fails | `JOBPDF_MONITOR_API_KEY` and the hash in `JOBPDF_API_KEYS` |

A CV that cannot be read (corrupt, encrypted, no text, too large) is the caller's problem and never
counts as an error here. The rates need at least 10 analyses and cover the time since the app last
restarted (metrics live in memory).

**Noise control.** An alert starts only after it was seen twice in a row (two minutes), so one slow
answer wakes nobody; it is repeated every 6 hours while it lasts; it is resolved after two clean
checks. A message that cannot be posted is retried every minute. The state is kept in
`data/monitor_state.json`. The numbers are constants at the top of `jobpdf/monitoring/alerts.py` and
`jobpdf/monitoring/uptime.py`.

**Ignore what does not apply.** A site without Airflow reports `check.fetch.no_cache` forever. Put
`MONITOR_IGNORE=check.fetch.no_cache` in `.env` (comma-separated) and `docker compose up -d monitor`.

**Without Docker, or from another machine.** The script is stand-alone:

```bash
uv run python scripts/uptime_check.py --url https://jobs.example.com      # once; exit 0 ok, 1 warning, 2 critical
# cron, every minute, with memory between runs (confirmation and the webhook need --state):
* * * * * cd /srv/jobpdf && ALERT_WEBHOOK_URL=... uv run python scripts/uptime_check.py --state data/monitor_state.json >/dev/null
```

**What the monitor cannot see.** It runs on the same machine as the app. If the machine, the Docker
daemon or the proxy (Caddy, the certificate) goes down, the monitor goes with it or looks at the wrong
layer. Add one outside check on `https://<DOMAIN>/health` (a free uptime service, or this script from
another machine with `--url`): `/health` is public, answers 200 for ok and degraded and 503 for down.
Nothing watches the monitor itself; `docker compose ps` shows whether it is running.
