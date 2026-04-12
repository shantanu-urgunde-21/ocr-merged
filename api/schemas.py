"""
API Request/Response Schemas using Pydantic
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    """Request model for OCR endpoint."""
    
    model_type: Literal["crnn", "tesseract"] = Field(
        default="crnn",
        description="Which model to use for OCR"
    )
    
    preprocessing_mode: Literal["full", "single_line"] = Field(
        default="full",
        description="'full' = deskew + line segmentation, 'single_line' = direct to model"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "crnn",
                "preprocessing_mode": "full"
            }
        }


class LineResult(BaseModel):
    """Result for a single recognized line."""
    
    line_index: int = Field(..., description="Index of this line")
    text: str = Field(..., description="Recognized text")
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score if available (CRNN doesn't provide this)"
    )


class FileResult(BaseModel):
    """OCR result for a single file."""
    
    file_index: int = Field(..., description="Index of uploaded file")
    filename: str = Field(..., description="Original filename")
    text: str = Field(..., description="Full concatenated text from file")
    lines: List[LineResult] = Field(..., description="Individual line results")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class OCRResponse(BaseModel):
    """Response model for OCR endpoint."""
    
    num_files: int = Field(..., description="Number of files processed")
    
    results: List[FileResult] = Field(..., description="Per-file OCR results")
    
    concatenated_text: str = Field(
        ...,
        description="All recognized text concatenated with newlines"
    )
    
    concatenated_lines: List[str] = Field(
        ...,
        description="All recognized lines across all files"
    )
    
    model_used: str = Field(
        ...,
        description="Which model was used for this request"
    )
    
    preprocessing_mode: str = Field(
        ...,
        description="Which preprocessing mode was used"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "num_files": 1,
                "results": [
                    {
                        "file_index": 0,
                        "filename": "example.png",
                        "text": "Hello World",
                        "lines": [
                            {
                                "line_index": 0,
                                "text": "Hello World",
                                "confidence": None
                            }
                        ],
                        "error": None
                    }
                ],
                "concatenated_text": "Hello World",
                "concatenated_lines": ["Hello World"],
                "model_used": "crnn",
                "preprocessing_mode": "full"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Status of the service")
    models_available: List[str] = Field(..., description="Available OCR models")
    device: str = Field(..., description="Compute device (cpu/cuda)")


class ModelInfoResponse(BaseModel):
    """Information about available models."""
    
    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable name")
    best_for: str = Field(..., description="Best use case")
    languages: str = Field(..., description="Supported languages")
    vocab_size: str = Field(..., description="Vocabulary size")
    input_size: str = Field(..., description="Expected input dimensions")


class ModelsInfoResponse(BaseModel):
    """Response with information about all available models."""
    
    available_models: List[str] = Field(..., description="List of model IDs")
    models: dict = Field(..., description="Detailed info per model")
