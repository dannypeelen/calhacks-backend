FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENV=production \
    SENTRIAI_TMP=/app/.tmp

# System deps (OpenCV runtime libs, build tools for wheels when needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better Docker layer caching
COPY requirements-prod.txt ./
RUN python -m pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . /app

# Prepare temp directory used by the app
RUN mkdir -p ${SENTRIAI_TMP}

# Create non-root user and set permissions
RUN adduser --disabled-password --gecos "" --uid 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default command: run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
