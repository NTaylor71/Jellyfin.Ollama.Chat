FROM python:3.13-slim

# Install required tools (curl for health checks) with minimal footprint
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only pyproject.toml first to maximize Docker cache
COPY pyproject.toml ./
COPY src src

# Install project dependencies using pip cache optimization
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir .

# Run the ingestion entrypoint
ENTRYPOINT ["./docker/faiss_ingestor/entrypoint.sh"]
