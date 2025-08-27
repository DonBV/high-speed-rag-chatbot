# Dockerfile
FROM python:3.11-slim

# System deps (optional, but good for ssl/certs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# App code
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
