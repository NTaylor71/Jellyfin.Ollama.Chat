FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for FAISS
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies - ONLY FAISS dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[faiss]"

# Create directories for FAISS data
RUN mkdir -p /app/data/faiss_index && \
    mkdir -p /app/logs

# Create non-root user
RUN groupadd -r faiss && useradd -r -g faiss faiss && \
    chown -R faiss:faiss /app
USER faiss

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:6333/health || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FAISS_INDEX_PATH=/app/data/faiss_index

# Expose port
EXPOSE 6333

# Start the FAISS service
CMD ["python", "-m", "src.faiss_service.main"]
