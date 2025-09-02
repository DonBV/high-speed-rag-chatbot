# High-Speed RAG API (FastAPI + Postgres/pgvector)

[![CodeQL](https://github.com/DonBV/high-speed-rag-chatbot/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/DonBV/high-speed-rag-chatbot/actions/workflows/codeql.yml)
[![pre-commit.ci](https://results.pre-commit.ci/badge/github/DonBV/high-speed-rag-chatbot/main.svg)](https://results.pre-commit.ci/latest/github/DonBV/high-speed-rag-chatbot/main)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)

Minimal REST API for semantic search: embeddings via OpenAI, storage in **Postgres + pgvector** with an **HNSW** index, and ranking with the `<=>` distance operator. One-command local start with Docker Compose.
Interactive API docs live at **`/docs`** (Swagger UI) and **`/redoc`** (ReDoc).

---

## Features

* Store embeddings in Postgres (`vector(n)`) and search with an **HNSW** ANN index
* Simple FastAPI endpoints: `POST /ingest` (upsert doc + embedding), `POST /search` (k-NN by cosine distance)
* One-command local deployment with **`docker compose`** (Compose v2)
* `.env.example` for quick configuration

---

## Project layout

```
app/
  __init__.py
  main.py               # FastAPI: /ingest, /search
deployment/
  init/02_schema.sql    # CREATE EXTENSION vector; documents table; HNSW index
Dockerfile              # API image
docker-compose.yml      # db (postgres+pgvector) + api
requirements-api.txt    # API deps
.env.example            # environment template
```

---

## Quickstart

### 1) Environment variables

Copy the template and fill in your keys:

```

cp .env.example .env

```

Key variables:

| Variable         | Purpose                       | Example / default                            |
| ---------------- | ----------------------------- | -------------------------------------------- |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | `sk-...`                                     |
| `DATABASE_URL`   | Postgres connection string    | `postgresql://postgres:postgres@db:5432/rag` |
| `EMBED_MODEL`    | Embedding model               | `text-embedding-3-small` (**1536 dims**)     |

> The `vector(n)` column dimension **must match** the embedding model’s output dimension (e.g., 1536 for `text-embedding-3-small`).

> **Windows (PowerShell)**
>
> ```
> Copy-Item .env.example .env
> # Ingest
> Invoke-RestMethod -Uri http://localhost:8000/ingest -Method Post -ContentType 'application/json' -Body '{"id":1,"content":"FastAPI is a modern web framework for building APIs"}'
> # Search
> Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType 'application/json' -Body '{"query":"vector search in postgres","k":3}'
> ```

### 2) Start the stack

```
docker compose up -d
```

* On **first start**, Postgres automatically runs any `*.sql` in `/docker-entrypoint-initdb.d/` — our schema and HNSW index will be created.
* If you previously started the DB and need a clean init, run `docker compose down -v` first, then `up -d`.
  *(Init scripts run **only** on an empty data directory / fresh volume.)*

> **Compose note**: On older setups that still use Compose v1, the command is `docker-compose up -d`. Newer Docker uses **Compose v2** with `docker compose`. 

Docs:

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
 
### 3) Smoke test (cURL)

Ingest a couple of docs:

```
curl -s http://localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{ "id": 1, "content": "FastAPI is a modern web framework for building APIs" }'

curl -s http://localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{ "id": 2, "content": "Postgres with pgvector enables vector similarity search" }'
```

Search:

```
curl -s http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{ "query": "vector search in postgres", "k": 3 }' | jq .
```

*(Optional)* If `jq` is not installed, remove `| jq .` to print raw JSON.

Expected:

```json
{
  "hits": [
    { "id": 2, "content": "Postgres with pgvector enables vector similarity search", "distance": 0.12 }
  ]
}
```

---

## API

> Base URL (local): `http://localhost:8000`

| Method | Path      | Request JSON                                         | 200 Response JSON                                                            | Description                                                                       |
| -----: | --------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
|   POST | `/ingest` | `{ "content": "string", "id": 123 }` *(id optional)* | `{ "id": 123 }`                                                              | Computes embedding (OpenAI) and upserts into Postgres/pgvector.                   |
|   POST | `/search` | `{ "query": "string", "k": 3 }`                      | `{ "hits": [ { "id": int, "content": "string", "distance": float }, ... ] }` | Returns top-k nearest neighbors by cosine distance (`ORDER BY embedding <=> qv`). |
|    GET | `/docs`   | —                                                    | Swagger UI                                                                   | Auto-docs (FastAPI).                                                              |
|    GET | `/redoc`  | —                                                    | ReDoc                                                                        | Alternative docs UI.                                                              |

---

## How it works

* **Embeddings**: the API calls OpenAI’s embeddings endpoint (default model `text-embedding-3-small`) and stores the resulting vector in Postgres as a pgvector literal like `[v1,v2,...]`. *(Default dimension 1536; keep DB column in sync.)* 
* **Postgres + pgvector**: the table has a `vector(1536)` column (matching the default model). We build an **HNSW** index with `vector_cosine_ops` for fast ANN search and order results with the cosine-distance operator `<=>`. 
* **Connection pooling**: `psycopg_pool.ConnectionPool` keeps DB connections warm to reduce latency and support concurrency.

---

## Local development (no Docker)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
export $(grep -v '^#' .env | xargs)     # inject env vars
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Troubleshooting

* **`type "vector" does not exist`** — the extension wasn’t installed: you likely started with a non-empty volume. Either run `CREATE EXTENSION vector;` manually or do `docker compose down -v` and `up -d` again. *(Init scripts run only on a fresh data directory.)* 
* **Init script didn’t run** — scripts in `/docker-entrypoint-initdb.d` run **only** on first initialization; if the volume isn’t empty, they’re skipped. Use `docker compose down -v` to reset volumes, then `up -d`.
* **`dimension mismatch`** — your `vector(n)` column doesn’t match the embedding dimension of the chosen model (e.g., 1536 by default for `text-embedding-3-small`). 
* **OpenAI errors** — ensure `OPENAI_API_KEY` is set and the API container can reach the internet.

---

## Security

* Do **not** commit `.env` (it’s ignored by `.gitignore`).
* Keep secrets in local env or CI secrets.
* Repo is wired with Code Scanning (CodeQL) and pre-commit CI.

---

## Credits

This repo was inspired by the AWS Solutions sample **“guidance-for-high-speed-rag-chatbots-on-aws”** (MIT-0). See the upstream for the full cloud-native reference.

---

## License

MIT-0 — see [LICENSE](./LICENSE).

---

