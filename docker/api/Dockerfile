FROM python:3.13-slim

WORKDIR /app

# Install basic tools with minimal footprint
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl redis-tools && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies using Docker BuildKit cache
COPY pyproject.toml .

COPY src src

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir -e .[dev]

# Default command to run the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

