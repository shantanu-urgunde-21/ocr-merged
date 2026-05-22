"""
CRNN-Specific Image Preprocessing & Normalization
Transforms images to match CRNN training expectations
Input: grayscale line images (64×512)
"""

import numpy as np
from PIL import Image
from typing import Optional, Any

try:
    import torch
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Match CRNN training configuration
from models.constants import IMG_HEIGHT, IMG_WIDTH


# Inference/validation transforms (only defined if PyTorch is available)
if HAS_TORCH:
    inference_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomAffine(
            degrees=2,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
            shear=2
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
else:
    inference_transform = None
    train_transform = None
    val_transform = None


class CRNNImagePreprocessor:
    """
    Preprocessing utilities for CRNN model.
    Handles numpy arrays, PIL images, and file paths.
    Supports both PyTorch and pure NumPy output backends.
    """
    
    @staticmethod
    def preprocess_pil_to_numpy(image: Image.Image) -> np.ndarray:
        """
        Pure NumPy/PIL preprocessing matching torchvision inference_transform.
        
        Args:
            image: PIL Image object
            
        Returns:
            Float32 NumPy array of shape (1, 1, 64, 512) normalized to [-1.0, 1.0]
        """
        # Resize to (IMG_WIDTH, IMG_HEIGHT) i.e. (512, 64) with BILINEAR resampling and convert to grayscale ('L')
        img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR).convert('L')
        
        # Convert to float32 NumPy array
        arr = np.array(img_resized, dtype=np.float32)
        
        # Scale to [0.0, 1.0] (equivalent to transforms.ToTensor())
        arr /= 255.0
        
        # Normalize with mean=0.5, std=0.5: (x - 0.5) / 0.5 = 2.0 * x - 1.0
        arr = (arr - 0.5) / 0.5
        
        # Add batch and channel dimensions (1, 1, 64, 512)
        arr = np.expand_dims(arr, axis=(0, 1))
        
        return arr

    @staticmethod
    def from_pil_numpy(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to model input NumPy array.
        
        Args:
            image: PIL Image object
            
        Returns:
            Float32 NumPy array of shape (1, 1, 64, 512)
        """
        return CRNNImagePreprocessor.preprocess_pil_to_numpy(image)

    @staticmethod
    def from_numpy_numpy(image: np.ndarray) -> np.ndarray:
        """
        Convert NumPy array to model input NumPy array.
        
        Args:
            image: NumPy array (grayscale or RGB)
            
        Returns:
            Float32 NumPy array of shape (1, 1, 64, 512)
        """
        if len(image.shape) == 2:  # Grayscale
            pil_img = Image.fromarray(image, mode='L')
        else:
            pil_img = Image.fromarray(image.astype(np.uint8), mode='RGB')
        return CRNNImagePreprocessor.preprocess_pil_to_numpy(pil_img)

    @staticmethod
    def from_numpy(
        image: np.ndarray,
        mode: str = "inference"
    ) -> Any:
        """
        Convert numpy array to model input tensor (PyTorch).
        """
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not available. Install PyTorch to use from_numpy(), "
                "or use from_numpy_numpy() to get pure NumPy preprocessed arrays."
            )
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = Image.fromarray(image, mode='L')
            else:
                image = Image.fromarray(image.astype(np.uint8), mode='RGB')
        
        transform = train_transform if mode == "training" else inference_transform
        tensor = transform(image)
        return tensor.unsqueeze(0)
    
    @staticmethod
    def from_pil(
        image: Image.Image,
        mode: str = "inference"
    ) -> Any:
        """
        Convert PIL Image to model input tensor (PyTorch).
        """
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not available. Install PyTorch to use from_pil(), "
                "or use from_pil_numpy() to get pure NumPy preprocessed arrays."
            )
        transform = train_transform if mode == "training" else inference_transform
        tensor = transform(image)
        return tensor.unsqueeze(0)
    
    @staticmethod
    def from_file(
        image_path: str,
        mode: str = "inference"
    ) -> Any:
        """
        Load image from file and convert to model input tensor (PyTorch).
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available. Install PyTorch to use from_file().")
        image = Image.open(image_path).convert("RGB")
        transform = train_transform if mode == "training" else inference_transform
        tensor = transform(image)
        return tensor.unsqueeze(0)
    
    @staticmethod
    def batch_from_numpy(
        images: list,
        mode: str = "inference"
    ) -> Any:
        """
        Convert list of numpy arrays to batch tensor (PyTorch).
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available. Install PyTorch to use batch_from_numpy().")
        transform = train_transform if mode == "training" else inference_transform
        tensors = []
        
        for img in images:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 2:
                    img = Image.fromarray(img, mode='L')
                else:
                    img = Image.fromarray(img.astype(np.uint8), mode='RGB')
            
            tensor = transform(img)
            tensors.append(tensor)
        
        return torch.stack(tensors)


# Utility functions for preprocessing pipeline

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if len(image.shape) == 2:
        return Image.fromarray(image, mode='L')
    else:
        return Image.fromarray(image.astype(np.uint8))


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(image)


def standardize_image_shape(
    image: np.ndarray,
    target_height: int = IMG_HEIGHT
) -> np.ndarray:
    """
    Resize image to standard CRNN input height while maintaining aspect ratio.
    """
    if len(image.shape) != 2:
        raise ValueError("Expected grayscale image (2D array)")
    
    h, w = image.shape
    scale = target_height / h
    new_w = int(w * scale)
    
    import cv2
    return cv2.resize(image, (new_w, target_height))
