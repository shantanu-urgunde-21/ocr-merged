import io
from pathlib import Path
from typing import List, Tuple, Optional, Any
import numpy as np
from PIL import Image

try:
    import torch
    from models import predict_text
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from models import ModelRegistry, VOCAB
from preprocessing import ImagePreprocessor, CRNNImagePreprocessor

from .schemas import FileResult, LineResult, OCRResponse


class OCRProcessor:
    """
    High-level OCR processing logic.
    Handles model selection, preprocessing, and result formatting.
    """
    
    def __init__(self, crnn_model_path: str, device: Optional[Any] = None):
        """
        Initialize processor with model paths.
        
        Args:
            crnn_model_path: Path to CRNN checkpoint file
            device: Custom device (auto-detected if None)
        """
        self.crnn_model_path = crnn_model_path
        if device is None:
            if HAS_TORCH:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.image_preprocessor = ImagePreprocessor()
        self.crnn_preprocessor = CRNNImagePreprocessor()
    
    def process_image_bytes(
        self,
        image_bytes: bytes,
        filename: str,
        model_type: str = "crnn",
        preprocessing_mode: str = "full"
    ) -> Tuple[FileResult, List[str]]:
        """
        Process a single image from bytes.
        
        Args:
            image_bytes: Raw image data
            filename: Original filename
            model_type: "crnn" or "tesseract"
            preprocessing_mode: "full" or "single_line"
            
        Returns:
            Tuple of (FileResult, list of line texts)
        """
        text_lines = []
        error = None
        
        try:
            if preprocessing_mode == "full":
                text_lines = self._process_full_pipeline(
                    image_bytes, model_type
                )
            else:  # single_line
                text_lines = self._process_single_line(
                    image_bytes, model_type
                )
        
        except Exception as e:
            error = str(e)
            text_lines = []
        
        # Format results
        line_results = [
            LineResult(line_index=i, text=text)
            for i, text in enumerate(text_lines)
        ]
        
        concatenated = "\n".join(text_lines)
        
        file_result = FileResult(
            file_index=0,
            filename=filename,
            text=concatenated,
            lines=line_results,
            error=error
        )
        
        return file_result, text_lines
    
    def _process_full_pipeline(
        self,
        image_bytes: bytes,
        model_type: str
    ) -> List[str]:
        """
        Full preprocessing pipeline: deskew -> segment lines -> predict each.
        
        Args:
            image_bytes: Raw image data
            model_type: "crnn" or "tesseract"
            
        Returns:
            List of recognized text lines
        """
        # Preprocess to get line images
        line_images = self.image_preprocessor.process_bytes(image_bytes)
        
        if not line_images:
            return []
        
        text_lines = []
        
        if model_type == "crnn":
            text_lines = self._predict_crnn_lines(line_images)
        elif model_type == "crnn_onnx":
            text_lines = self._predict_onnx_lines(line_images)
        elif model_type == "tesseract":
            text_lines = self._predict_tesseract_lines(line_images)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        return text_lines
    
    def _process_single_line(
        self,
        image_bytes: bytes,
        model_type: str
    ) -> List[str]:
        """
        Single-line mode: treat image as single line, apply CRNN preprocessing directly.
        Skips segmentation - assumes input is already a single line.
        
        Args:
            image_bytes: Raw image data (should be single line)
            model_type: "crnn" or "tesseract"
            
        Returns:
            List with single recognized line (or empty if failed)
        """
        # Decode image
        arr = np.frombuffer(image_bytes, np.uint8)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if model_type == "crnn":
            tensor = self.crnn_preprocessor.from_pil(image, mode="inference")
            model = ModelRegistry.get_model(
                "crnn",
                device=self.device,
                model_path=self.crnn_model_path
            )
            text = predict_text(model, tensor, self.device)
            return [text] if text else []
        
        elif model_type == "crnn_onnx":
            numpy_arr = self.crnn_preprocessor.from_pil_numpy(image)
            model = ModelRegistry.get_model(
                "crnn_onnx",
                device=self.device,
                model_path=self.crnn_model_path
            )
            text = model.predict(numpy_arr)
            return [text] if text else []
        
        elif model_type == "tesseract":
            # Convert to numpy for Tesseract
            img_array = np.array(image.convert("L"))
            model = ModelRegistry.get_model("tesseract")
            text = model.predict(img_array)
            return [text] if text else []
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _predict_crnn_lines(self, line_images: List[np.ndarray]) -> List[str]:
        """
        Run CRNN inference on preprocessed line images.
        
        Args:
            line_images: List of grayscale line images
            
        Returns:
            List of recognized text strings
        """
        model = ModelRegistry.get_model(
            "crnn",
            device=self.device,
            model_path=self.crnn_model_path
        )
        
        text_lines = []
        for line_img in line_images:
            try:
                tensor = self.crnn_preprocessor.from_numpy(line_img, mode="inference")
                text = predict_text(model, tensor, self.device)
                if text:  # Only append non-empty results
                    text_lines.append(text)
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
        
        return text_lines
    
    def _predict_onnx_lines(self, line_images: List[np.ndarray]) -> List[str]:
        """
        Run ONNX CRNN inference on preprocessed line images.
        
        Args:
            line_images: List of grayscale line images
            
        Returns:
            List of recognized text strings
        """
        model = ModelRegistry.get_model(
            "crnn_onnx",
            device=self.device,
            model_path=self.crnn_model_path
        )
        
        text_lines = []
        for line_img in line_images:
            try:
                numpy_arr = self.crnn_preprocessor.from_numpy_numpy(line_img)
                text = model.predict(numpy_arr)
                if text:  # Only append non-empty results
                    text_lines.append(text)
            except Exception as e:
                print(f"Error processing ONNX line: {str(e)}")
                continue
        
        return text_lines
    
    def _predict_tesseract_lines(self, line_images: List[np.ndarray]) -> List[str]:
        """
        Run Tesseract inference on preprocessed line images.
        
        Args:
            line_images: List of grayscale line images
            
        Returns:
            List of recognized text strings
        """
        model = ModelRegistry.get_model("tesseract")
        
        text_lines = []
        for line_img in line_images:
            try:
                text = model.predict(line_img)
                if text:  # Only append non-empty results
                    text_lines.append(text)
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
        
        return text_lines


async def process_ocr_request(
    files: List,  # FastAPI UploadFile objects
    processor: OCRProcessor,
    model_type: str = "crnn",
    preprocessing_mode: str = "full"
) -> OCRResponse:
    """
    Process multiple files through OCR pipeline.
    
    Args:
        files: List of FastAPI UploadFile objects
        processor: Initialized OCRProcessor
        model_type: Model choice
        preprocessing_mode: Preprocessing strategy
        
    Returns:
        OCRResponse with results from all files
    """
    results = []
    all_lines = []
    
    for file_idx, file in enumerate(files):
        file_bytes = await file.read()
        
        file_result, lines = processor.process_image_bytes(
            file_bytes,
            file.filename,
            model_type=model_type,
            preprocessing_mode=preprocessing_mode
        )
        
        # Update file index
        file_result.file_index = file_idx
        results.append(file_result)
        all_lines.extend(lines)
    
    # Build response
    response = OCRResponse(
        num_files=len(files),
        results=results,
        concatenated_text="\n".join(all_lines),
        concatenated_lines=all_lines,
        model_used=model_type,
        preprocessing_mode=preprocessing_mode
    )
    
    return response
