FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (needed for setup.py/pyproject.toml)
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies - ONLY worker dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[worker]"

# Create directories
RUN mkdir -p /app/logs /app/data

# Create non-root user
RUN groupadd -r worker && useradd -r -g worker worker && \
    chown -R worker:worker /app

# Switch to non-root user
USER worker

# Health check - test Redis connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.redis_worker.queue_manager import RedisQueueManager; RedisQueueManager().redis_client.ping()" || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start the worker
CMD ["python", "-m", "src.redis_worker.main"]
