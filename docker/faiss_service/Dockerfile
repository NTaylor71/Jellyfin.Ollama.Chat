# /docker/faiss_service/Dockerfile

FROM python:3.13-slim

# Install curl and ping for health checks and connectivity testing
RUN apt-get update && \
    apt-get install -y curl iputils-ping && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy metadata first for layer caching
COPY pyproject.toml ./

# Copy source code (PEP 621 + /src layout)
COPY src/ ./src/

# Copy entrypoint and wait script
COPY docker/faiss_service/entrypoint.sh ./docker/faiss_service/entrypoint.sh
COPY docker/wait_for_services.sh ./wait_for_services.sh

# Make scripts executable
RUN chmod +x ./docker/faiss_service/entrypoint.sh && chmod +x ./wait_for_services.sh

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir .

# Set entrypoint
ENTRYPOINT ["./docker/faiss_service/entrypoint.sh"]
