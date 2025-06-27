FROM python:3.13-slim

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only what's needed for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies (PEP 621 + hatch)
RUN pip install --upgrade pip && pip install --no-cache-dir .

# Set entrypoint
ENTRYPOINT ["./docker/redis_queue_worker/entrypoint.sh"]
