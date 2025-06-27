FROM python:3.13-slim

# Install required tools (curl for health checks)
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first for layer caching
COPY pyproject.toml ./

# Copy source code (PEP 621 + /src layout)
COPY src/ ./src/

# Install project dependencies using Hatchling and pyproject.toml
RUN pip install --upgrade pip && pip install --no-cache-dir .

# Run the ingestion entrypoint
ENTRYPOINT ["./docker/faiss_ingestor/entrypoint.sh"]
