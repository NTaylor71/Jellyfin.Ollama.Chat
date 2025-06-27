FROM python:3.13-slim

# Install curl with minimal footprint
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only pyproject.toml first for layer caching
COPY pyproject.toml ./
COPY src src

# Install Python dependencies using pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir .

# Run the GPU worker entrypoint
ENTRYPOINT ["./docker/gpu_queue_worker/entrypoint.sh"]
