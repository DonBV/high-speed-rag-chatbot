-- deployment/init/02_schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Important: the vector size must match the selected embedding model.
-- text-embedding-3-small -> 1536
CREATE TABLE IF NOT EXISTS documents (
  id        BIGSERIAL PRIMARY KEY,
  content   TEXT NOT NULL,
  embedding vector(1536) NOT NULL
);

-- HNSW index under cosine "distance"
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
  ON documents
  USING hnsw (embedding vector_cosine_ops);
