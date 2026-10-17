# JobPDF images (JM-36). One file, three targets:
#
#   app      the API and the one-off scripts (scripts/*.py)       docker compose build app
#   airflow  Airflow with the jobpdf package and the DAGs         docker compose --profile airflow build
#
# The package is run from source in /app, not installed: it finds sql/ and data/ relative to
# its own folder (REPO_ROOT), so the working directory must be /app.

# ---- dependencies -----------------------------------------------------------------------
FROM python:3.12-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app

# Only the dependency list is copied first, so code changes do not reinstall torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
 && uv export --locked --no-dev --no-hashes --no-emit-project -o /opt/requirements.txt


# ---- app --------------------------------------------------------------------------------
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    HF_HOME=/opt/hf \
    HOME=/tmp

# PyMuPDF and torch need the OpenMP runtime; nothing else is installed from apt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=deps /opt/venv /opt/venv
COPY --from=deps /opt/requirements.txt /opt/requirements.txt
WORKDIR /app
COPY jobpdf ./jobpdf
COPY scripts ./scripts
COPY sql ./sql
COPY dags ./dags

# The embedding model is part of the image (about 470 MB), so a server never downloads it
# while starting and runs with HF_HUB_OFFLINE=1. The revision is pinned in the code.
RUN python -c "from huggingface_hub import snapshot_download; \
from jobpdf.normalization.embedder import MODEL_NAME, MODEL_REVISION; \
snapshot_download(MODEL_NAME, revision=MODEL_REVISION)" \
 && chmod -R a+rX /opt/hf /app

ENV HF_HUB_OFFLINE=1
EXPOSE 8000

# One worker: the model is loaded once and requests run in threads. The rate-limit counters
# live in Postgres, so more workers would still share the limits. No access log: it prints
# client addresses, which the limiter deliberately never stores.
CMD ["uvicorn", "jobpdf.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log"]


# ---- airflow ----------------------------------------------------------------------------
# Tested against Airflow 2.10 (the DAGs also import under Airflow 3).
FROM apache/airflow:2.10.5-python3.12 AS airflow

ENV PYTHONPATH=/app \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    AIRFLOW__CORE__DAGS_FOLDER=/app/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=false

USER root
# Torch and PyMuPDF need the OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=app /app /app
COPY --from=app /opt/hf /opt/hf
COPY --from=deps /opt/requirements.txt /opt/requirements.txt
RUN chmod -R a+rX /app /opt/hf /opt/requirements.txt
USER airflow

# The project's locked dependencies go into Airflow's own environment.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /opt/requirements.txt
