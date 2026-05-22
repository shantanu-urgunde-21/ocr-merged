FROM python:3.10-slim

# Install system dependencies
# Tesseract: OCR engine
# OpenGL: For image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Accept a build argument (cpu, gpu, or none) - default to none for ultra-lightweight ONNX-only runtime
ARG DEVICE_TYPE=none

# Install PyTorch dynamically based on build argument to save space/time
RUN if [ "$DEVICE_TYPE" = "cpu" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$DEVICE_TYPE" = "gpu" ]; then \
        pip install --no-cache-dir torch torchvision; \
    else \
        echo "Skipping PyTorch installation (using lightweight ONNX Runtime exclusively)"; \
    fi

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
