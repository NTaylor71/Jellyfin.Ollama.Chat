FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Java for SUTime
RUN apt-get update && apt-get install -y \
    curl \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for SUTime temporal parsing
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy project files (needed for setup.py/pyproject.toml)
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies - ONLY worker dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[worker]"

# Create directories for NLP models and data
RUN mkdir -p /app/logs /app/data /app/nlp_models

# Pre-download commonly used NLP models to prevent runtime downloads
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nlp_models'); nltk.download('stopwords', download_dir='/app/nlp_models'); nltk.download('wordnet', download_dir='/app/nlp_models')" || true

# Set NLTK data path
ENV NLTK_DATA=/app/nlp_models

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
