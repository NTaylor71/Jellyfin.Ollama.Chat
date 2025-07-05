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

# Install Python dependencies - ONLY API dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api]"

# Create directories
RUN mkdir -p /app/logs /app/data

# Create non-root user
RUN groupadd -r api && useradd -r -g api api && \
    chown -R api:api /app

# Switch to non-root user
USER api

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Start the API
CMD ["python", "-m", "src.api.main"]
