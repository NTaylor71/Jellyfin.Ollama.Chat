FROM python:3.13-slim

# Install curl and ping with minimal footprint
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl iputils-ping && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only pyproject.toml first for layer caching
COPY pyproject.toml ./
COPY src src

# Install Python dependencies using pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir .

# Copy entrypoint and wait script
COPY docker/faiss_service/entrypoint.sh ./docker/faiss_service/entrypoint.sh
COPY docker/wait_for_services.sh ./wait_for_services.sh

# Make scripts executable
RUN chmod +x ./docker/faiss_service/entrypoint.sh && chmod +x ./wait_for_services.sh

# Set entrypoint
ENTRYPOINT ["./docker/faiss_service/entrypoint.sh"]
