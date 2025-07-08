# =============================================================================
# MULTI-STAGE BUILD: Reduce final image size from 7.24GB to ~2GB
# =============================================================================

# =============================================================================
# STAGE 1: BUILDER - Install all dependencies and build tools
# =============================================================================
FROM python:3.12-slim as builder

# Install build dependencies (needed for compiling packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    jq \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for building
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy project files (needed for setup.py/pyproject.toml)
COPY pyproject.toml ./
COPY src/ ./src/

# Install all Python dependencies to /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[worker,mongodb,plugins,ollama,api]"

# =============================================================================
# STAGE 2: RUNTIME - Copy only necessary files from builder
# =============================================================================
FROM python:3.12-slim as runtime

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for runtime (JRE, not JDK)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml ./

# Create directories for data
RUN mkdir -p /app/logs /app/data

# Create non-root user
RUN groupadd -r worker && useradd -r -g worker worker && \
    chown -R worker:worker /app

# Switch to non-root user
USER worker

# Health check - test Redis connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.worker.resource_queue_manager import ResourceAwareQueueManager; from src.worker.resource_manager import create_resource_pool_from_config; import asyncio; pool = create_resource_pool_from_config({'cpu_cores': 1, 'gpu_count': 0, 'memory_mb': 512}, 'health'); asyncio.run(ResourceAwareQueueManager(pool).health_check())" || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start the worker
CMD ["python", "-m", "src.worker.main"]