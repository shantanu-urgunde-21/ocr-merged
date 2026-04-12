"""
Generic Image Preprocessing Module
Handles: loading, deskewing, line segmentation, line splitting, basic preparation
Used for full-page OCR workflow
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def load_image(image_path: str, target_width: int = 2000) -> np.ndarray:
    """
    Load and resize image to standard width while maintaining aspect ratio.
    
    Args:
        image_path: Path to image file
        target_width: Target width for resizing (default 2000)
        
    Returns:
        Grayscale image as numpy array
        
    Raises:
        ValueError: If image cannot be loaded
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape
    if w != target_width:
        scale = target_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (target_width, new_h))

    return img


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Deskew image using minAreaRect and rotation.
    Corrects slight page tilting/rotation.
    
    Args:
        img: Input grayscale image
        
    Returns:
        Deskewed image
    """
    # Threshold and invert for contour detection
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to connect nearby pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find rotation angle
    coords = np.column_stack(np.where(dilated > 0))
    if len(coords) == 0:
        return img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Apply rotation if significant
    if 0.5 < abs(angle) < 15:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def get_lines(img: np.ndarray) -> List[np.ndarray]:
    """
    Segment image into individual text lines using horizontal projection.
    
    Args:
        img: Input grayscale image
        
    Returns:
        List of line images (each is a 2D array)
    """
    # Threshold and invert
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: sum pixel values per row
    row_sums = np.sum(binary, axis=1).astype(np.float32)
    row_sums = cv2.GaussianBlur(row_sums.reshape(-1, 1), (1, 15), 0).flatten()

    # Threshold to identify text rows
    threshold = 0.05 * np.percentile(row_sums, 90)
    is_text = row_sums > threshold

    # Extract contiguous text regions
    lines = []
    start = None
    for i, val in enumerate(is_text):
        if val and start is None:
            start = i
        elif not val and start is not None:
            # Only keep lines with minimum height
            if i - start > 15:
                lines.append(img[start:i, :])
            start = None

    if start is not None and len(is_text) - start > 15:
        lines.append(img[start : len(is_text), :])

    return lines


def split_line(line: np.ndarray, max_width: int = 1000) -> List[np.ndarray]:
    """
    Split wide text lines into chunks if they exceed max_width.
    Finds the best split point (whitespace area) near the middle.
    
    Args:
        line: Single line image
        max_width: Maximum line width before splitting
        
    Returns:
        List of line chunks (usually 1 or 2 items)
    """
    h, w = line.shape
    if w <= max_width:
        return [line]

    # Find best split point in middle region
    mid_start = int(w * 0.4)
    mid_end = int(w * 0.6)
    middle = line[:, mid_start:mid_end]

    col_sums = np.sum(middle, axis=0)
    split_pos = mid_start + np.argmin(col_sums)  # Find whitespace area

    return [line[:, :split_pos], line[:, split_pos:]]


def prepare_line(line: np.ndarray, height: int = 64) -> Optional[np.ndarray]:
    """
    Prepare a single line for model input.
    
    - Resize to fixed height while maintaining aspect ratio
    - Normalize pixel values
    - Add padding
    
    Args:
        line: Input line image
        height: Target height (default 64 matches CRNN training)
        
    Returns:
        Prepared line image, or None if line is too small/invalid
    """
    h, w = line.shape
    if h < 5 or w < 5:
        return None

    # Scale to target height
    scale = height / h
    new_w = int(w * scale)
    line = cv2.resize(line, (new_w, height))
    
    # Normalize intensity
    line = cv2.normalize(line, None, 0, 255, cv2.NORM_MINMAX)
    
    # Add padding (white space) around line
    line = cv2.copyMakeBorder(line, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    return line


class ImagePreprocessor:
    """
    High-level image preprocessing pipeline for full-page OCR.
    
    Usage:
        preprocessor = ImagePreprocessor()
        lines = preprocessor.process_image(image_path)
    """
    
    def __init__(self, target_width: int = 2000, line_height: int = 64, max_line_width: int = 1000):
        """
        Initialize preprocessor with configuration.
        
        Args:
            target_width: Initial image width for resizing
            line_height: Target height for each line
            max_line_width: Maximum width before splitting line
        """
        self.target_width = target_width
        self.line_height = line_height
        self.max_line_width = max_line_width
    
    def process_image(self, image_path: str) -> List[np.ndarray]:
        """
        Process a full-page image and return prepared line images.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of prepared line images ready for model inference
        """
        # Load and deskew
        img = load_image(image_path, self.target_width)
        img = deskew(img)
        
        # Segment into lines
        lines = get_lines(img)
        prepared_lines = []
        
        for line in lines:
            # Split if too wide
            chunks = split_line(line, self.max_line_width)
            
            for chunk in chunks:
                prepared = prepare_line(chunk, self.line_height)
                if prepared is not None:
                    prepared_lines.append(prepared)
        
        return prepared_lines
    
    def process_bytes(self, image_bytes: bytes) -> List[np.ndarray]:
        """
        Process image from bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            List of prepared line images
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Cannot decode image from bytes")
        
        # Resize to standard width
        h, w = img.shape
        if w != self.target_width:
            scale = self.target_width / w
            img = cv2.resize(img, (self.target_width, int(h * scale)))
        
        # Deskew
        img = deskew(img)
        
        # Segment and prepare
        lines = get_lines(img)
        prepared_lines = []
        
        for line in lines:
            chunks = split_line(line, self.max_line_width)
            for chunk in chunks:
                prepared = prepare_line(chunk, self.line_height)
                if prepared is not None:
                    prepared_lines.append(prepared)
        
        return prepared_lines
