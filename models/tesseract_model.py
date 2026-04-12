"""
Tesseract OCR Model Adapter
Wrapper around pytesseract for general-purpose OCR
"""

import cv2
import numpy as np
from typing import Optional
import pytesseract


class TesseractModel:
    """
    Wrapper around Tesseract OCR engine.
    
    Best for:
    - General printed text (not just handwriting)
    - Fallback when CRNN fails
    - Diverse character sets and languages
    """
    
    def __init__(self, config: str = "--psm 7"):
        """
        Initialize Tesseract model.
        
        Args:
            config: Tesseract configuration string
                   "--psm 7" = single text line (default)
                   "--psm 6" = single block
                   See tesseract docs for other options
        """
        self.config = config
        self._verify_installation()
    
    def _verify_installation(self) -> None:
        """Check that Tesseract is properly installed."""
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract is not installed. "
                "Install from: https://github.com/UB-Mannheim/tesseract/wiki"
            )
    
    def predict(self, line_image: np.ndarray) -> str:
        """
        Run OCR on a single line image.
        
        Args:
            line_image: Input image as numpy array (grayscale or color)
            
        Returns:
            Recognized text string
        """
        if line_image is None or line_image.size == 0:
            return ""
        
        try:
            # Ensure image is 8-bit
            if line_image.dtype != np.uint8:
                line_image = (line_image * 255).astype(np.uint8)
            
            # Run OCR
            text = pytesseract.image_to_string(line_image, config=self.config)
            return text.strip()
        except Exception as e:
            print(f"Tesseract error: {str(e)}")
            return ""
    
    def predict_batch(self, images: list) -> list:
        """
        Run OCR on multiple images.
        
        Args:
            images: List of image arrays
            
        Returns:
            List of recognized text strings
        """
        return [self.predict(img) for img in images]
