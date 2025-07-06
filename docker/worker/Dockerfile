FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Java for SUTime and jq for JSON parsing
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for SUTime temporal parsing
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy project files (needed for setup.py/pyproject.toml)
COPY pyproject.toml ./
COPY src/ ./src/
COPY manage_models.py ./

# Install Python dependencies - Worker plus MongoDB, plugins, and ollama dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[worker,mongodb,plugins,ollama,nlp]"

# Create directories for models and data
RUN mkdir -p /app/logs /app/data /app/nlp_models /app/models

# Copy entrypoint script
COPY docker/scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set model environment variables (will be set by entrypoint)
ENV NLTK_DATA=/app/models/nltk_data
ENV GENSIM_DATA_DIR=/app/models/gensim_data

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

# Use entrypoint for model management
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start the worker
CMD ["python", "-m", "src.redis_worker.main"]
