"""
Main FastAPI Application
Unified OCR API supporting CRNN and Tesseract models
"""

from pathlib import Path
from typing import List, Optional, Annotated
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from models import ModelRegistry
from .routes import OCRProcessor, process_ocr_request
from .schemas import (
    OCRResponse,
    HealthResponse,
    ModelsInfoResponse,
    ModelInfoResponse,
)


# Initialize FastAPI app
app = FastAPI(
    title="Unified OCR API",
    description="OCR service supporting CRNN (handwriting) and Tesseract (general text)",
    version="1.0.0"
)

# Global processor instance (lazy loaded)
processor: Optional[OCRProcessor] = None
CRNN_MODEL_PATH = "best_model_weights/handwriting_recognizer_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_processor() -> OCRProcessor:
    """Lazy-load processor instance (CRNN weights checked only when model_type is crnn)."""
    global processor
    if processor is None:
        processor = OCRProcessor(CRNN_MODEL_PATH, device=DEVICE)
    return processor


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    print("=" * 60)
    print("OCR Service Startup")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"CRNN Model Path: {CRNN_MODEL_PATH}")
    
    # Verify CRNN model exists
    if not Path(CRNN_MODEL_PATH).exists():
        print(f"⚠️  Warning: CRNN model not found at {CRNN_MODEL_PATH}")
        print("   CRNN inference will fail until model is available")
    else:
        print(f"✓ CRNN model found")
    
    # Verify Tesseract (optional)
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✓ Tesseract is installed")
    except:
        print("⚠️  Tesseract not installed - Tesseract model will not work")
    
    print("=" * 60)


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service status and available models."""
    return HealthResponse(
        status="healthy",
        models_available=ModelRegistry.get_available_models(),
        device=str(DEVICE)
    )


@app.get("/models", response_model=ModelsInfoResponse)
async def get_models_info():
    """Get detailed information about available models."""
    available = ModelRegistry.get_available_models()
    models_info = {}
    
    for model_type in available:
        info = ModelRegistry.get_model_info(model_type)
        models_info[model_type] = info
    
    return ModelsInfoResponse(
        available_models=available,
        models=models_info
    )


@app.get("/models/{model_type}", response_model=ModelInfoResponse)
async def get_model_info(model_type: str):
    """Get information about a specific model."""
    try:
        info = ModelRegistry.get_model_info(model_type)
        return ModelInfoResponse(
            model_id=model_type,
            **info
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# OCR ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API documentation root."""
    return {
        "message": "Unified OCR API",
        "endpoints": {
            "health_check": "/health",
            "models_info": "/models",
            "ocr_processing": "/ocr (POST), /ocr/single (POST)",
        },
        "docs": "/docs",
        "models_available": ModelRegistry.get_available_models()
    }


async def _run_ocr(
    files: List[UploadFile],
    model_type: str,
    preprocessing_mode: str,
) -> OCRResponse:
    """Shared validation + inference for /ocr and /ocr/single."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if model_type not in ModelRegistry.get_available_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type '{model_type}'. Available: {ModelRegistry.get_available_models()}",
        )

    if preprocessing_mode not in ["full", "single_line"]:
        raise HTTPException(
            status_code=400,
            detail="preprocessing_mode must be 'full' or 'single_line'",
        )

    allowed_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
    for uf in files:
        ext = Path(uf.filename or "").suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext!r}. Allowed: {allowed_extensions}",
            )

    if model_type == "crnn" and not Path(CRNN_MODEL_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"CRNN weights not found. Add handwriting_recognizer_best.pth under best_model_weights/ "
                f"(expected path: {CRNN_MODEL_PATH}). Tesseract does not require this file."
            ),
        )

    try:
        proc = get_processor()
        return await process_ocr_request(
            files,
            proc,
            model_type=model_type,
            preprocessing_mode=preprocessing_mode,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")




@app.post("/ocr", response_model=OCRResponse)
async def process_ocr(
    model_type: str = Query("crnn", description="'crnn' for handwritten text (default) or 'tesseract' for general text"),
    preprocessing_mode: str = Query("full", description="'full' for deskew + line segmentation or 'single_line' for direct inference"),
    files: Annotated[List[UploadFile], File(description="Select one or more image files. In Swagger, click 'Add item' then use the file picker.")] = [],
):
    """
    OCR one or more images. Supports multiple file uploads.
    If you find the 'Add item' list confusing in Swagger, use **POST /ocr/single** instead.
    """
    if files is None:
        raise HTTPException(status_code=400, detail="No files provided")
    return await _run_ocr(files, model_type, preprocessing_mode)


@app.post("/ocr/single", response_model=OCRResponse)
async def process_ocr_single(
    model_type: str = Query("crnn", description="'crnn' or 'tesseract'"),
    preprocessing_mode: str = Query("full", description="'full' or 'single_line'"),
    file: UploadFile = File(..., description="One image file"),
):
    """
    OCR a single image. Use this if the main /ocr endpoint's multi-file picker is confusing in Swagger UI.
    """
    return await _run_ocr([file], model_type, preprocessing_mode)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
