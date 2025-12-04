from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from openai import AsyncOpenAI  # <--- Async Client
from psycopg_pool import AsyncConnectionPool  # <--- Async Pool
from pydantic import BaseModel, Field

# --- config from env ---
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # <--- FIX 1: Base URL
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# --- Async Setup ---
pool = AsyncConnectionPool(conninfo=DATABASE_URL, max_size=10, open=False)
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL
)  # <--- FIX 2: Base URL


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()  # Open DB pool on startup
    yield
    await pool.close()  # Close on shutdown


app = FastAPI(title="High-Speed RAG API", version="0.1.0", lifespan=lifespan)


# --- pydantic models ---
class IngestRequest(BaseModel):
    content: str = Field(..., min_length=1)
    id: Optional[int] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=50)


class Hit(BaseModel):
    id: int
    content: str
    distance: float


class SearchOut(BaseModel):
    hits: List[Hit]


# --- Async Functions ---
async def embed_text(text: str) -> List[float]:
    # Await the API call
    res = await client.embeddings.create(model=EMBED_MODEL, input=text)
    return res.data[0].embedding


def to_pgvector(vec: List[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


@app.get("/")
async def root():  # Async def
    return {"ok": True, "model": EMBED_MODEL}


@app.post("/ingest")
async def ingest(item: IngestRequest):  # Async def
    pgv = to_pgvector(await embed_text(item.content))  # Await embedding
    async with pool.connection() as conn:  # Async connection
        async with conn.cursor() as cur:
            if item.id is None:
                await cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector) RETURNING id",
                    (item.content, pgv),
                )
                new_id = (await cur.fetchone())[0]
                await conn.commit()
                return {"id": new_id}
            else:
                await cur.execute(
                    """
                    INSERT INTO documents (id, content, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE
                      SET content = EXCLUDED.content,
                          embedding = EXCLUDED.embedding
                    RETURNING id
                    """,
                    (item.id, item.content, pgv),
                )
                upsert_id = (await cur.fetchone())[0]
                await conn.commit()
                return {"id": upsert_id}


@app.post("/search", response_model=SearchOut)
async def search(payload: SearchRequest) -> SearchOut:  # Async def
    qv = to_pgvector(await embed_text(payload.query))  # Await embedding
    sql = """
      SELECT id, content, (embedding <=> %s::vector) AS distance
      FROM documents
      ORDER BY embedding <=> %s::vector
      LIMIT %s
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (qv, qv, payload.k))
            rows = await cur.fetchall()
    return {"hits": [Hit(id=r[0], content=r[1], distance=float(r[2])) for r in rows]}
