FROM python:3.13-slim

WORKDIR /app

# Install basic tools
RUN apt-get update && \
    apt-get install -y curl redis-tools && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install -e .[dev]

# Copy source code
COPY src src

# Default command to run the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
