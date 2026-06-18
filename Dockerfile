FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Render uses $PORT env var; default 8000 for local)
ENV PORT=8000
EXPOSE $PORT

# Gunicorn + uvicorn workers for production concurrency
# Adjust --workers based on (2 × CPU cores) + 1 for your instance size
CMD uvicorn app.main:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers 2 \
        --loop uvloop \
        --access-log
