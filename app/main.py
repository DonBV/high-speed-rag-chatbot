# app/main.py
import os
from typing import List, Optional

from fastapi import FastAPI
from openai import OpenAI
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

# --- config from env ---
DATABASE_URL = os.environ[
    "DATABASE_URL"
]  # e.g. postgresql://postgres:postgres@db:5432/rag
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "text-embedding-3-small"
)  # 1536-dim by default

pool = ConnectionPool(conninfo=DATABASE_URL, max_size=10)
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="RAG Vector API", version="0.1.0")


# --- pydantic models ---
class IngestIn(BaseModel):
    content: str = Field(..., min_length=1)
    id: Optional[int] = None


class SearchIn(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=50)


class Hit(BaseModel):
    id: int
    content: str
    distance: float


class SearchOut(BaseModel):
    hits: List[Hit]


def embed_text(text: str) -> List[float]:
    # OpenAI embeddings
    res = client.embeddings.create(model=EMBED_MODEL, input=text)
    return res.data[0].embedding


def to_pgvector(vec: List[float]) -> str:
    # textual literal for pgvector: [v1,v2,...]
    return "[" + ",".join(str(x) for x in vec) + "]"


@app.get("/")
def root():
    return {"ok": True, "model": EMBED_MODEL}


@app.post("/ingest")
def ingest(item: IngestIn):
    pgv = to_pgvector(embed_text(item.content))
    with pool.connection() as conn, conn.cursor() as cur:
        if item.id is None:
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector) RETURNING id",
                (item.content, pgv),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            return {"id": new_id}
        else:
            cur.execute(
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
            upsert_id = cur.fetchone()[0]
            conn.commit()
            return {"id": upsert_id}


@app.post("/search", response_model=SearchOut)
def search(payload: SearchIn):
    qv = to_pgvector(embed_text(payload.query))
    # cosine distance operator in pgvector is <=> ; lower = more similar
    sql = """
      SELECT id, content, (embedding <=> %s::vector) AS distance
      FROM documents
      ORDER BY embedding <=> %s::vector
      LIMIT %s
    """
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (qv, qv, payload.k))
        rows = cur.fetchall()
    return {"hits": [Hit(id=r[0], content=r[1], distance=float(r[2])) for r in rows]}
