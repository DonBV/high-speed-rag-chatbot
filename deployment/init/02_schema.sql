-- Requires: deployment/init/01_pgvector.sql (CREATE EXTENSION vector)
-- Schema for 1536-dim embeddings (OpenAI `text-embedding-3-small`) + HNSW (cosine)

-- Main table for RAG chunks
CREATE TABLE IF NOT EXISTS documents (
  id         BIGSERIAL PRIMARY KEY,
  content    TEXT NOT NULL,              -- original text/snippet
  embedding  VECTOR(1536) NOT NULL,      -- 1536 for text-embedding-3-small
  source     TEXT,                       -- optional: path/URL/tag
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Approximate nearest-neighbor index for cosine similarity (recommended default)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_cos
  ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Tips:
-- 1) To increase recall at query time, bump the search breadth for the current session:
--    SET hnsw.ef_search = 80;  -- higher => better recall, slower (default is 40)
-- 2) After bulk loads, refresh planner stats:
--    ANALYZE documents;

-- Example top-K query (cosine distance: lower = closer):
-- SELECT id, content, (embedding <=> $QUERY_VECTOR) AS distance
-- FROM documents
-- ORDER BY embedding <=> $QUERY_VECTOR
-- LIMIT 5;
