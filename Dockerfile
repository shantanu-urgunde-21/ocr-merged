FROM python:3.10-slim

# Install system dependencies
# Tesseract: OCR engine
# OpenGL: For image processing
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libheif-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install API/runtime dependencies (weights mounted at runtime via compose)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

RUN mkdir -p model_weights

# Copy application code
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/
COPY api/ ./api/

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
