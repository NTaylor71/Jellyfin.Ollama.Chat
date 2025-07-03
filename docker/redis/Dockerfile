FROM redis:7-alpine

# Install additional tools for debugging (optional)
RUN apk add --no-cache \
    curl \
    netcat-openbsd

# Copy custom Redis configuration
COPY docker/redis/redis.conf /etc/redis/redis.conf

# Ensure data directory exists and has correct permissions
RUN mkdir -p /data && \
    chown redis:redis /data && \
    chmod 755 /data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD redis-cli ping || exit 1

# Switch to redis user (already exists in the base image)
USER redis

# Expose Redis port
EXPOSE 6379

# Start Redis with custom config
CMD ["redis-server", "/etc/redis/redis.conf"]
