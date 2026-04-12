"""
CRNN-Specific Image Preprocessing
Transforms images to match CRNN training expectations
Input: grayscale line images (64×512)
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional


# Match CRNN training configuration
IMG_HEIGHT = 64
IMG_WIDTH = 512


# Inference transform (no augmentation, deterministic)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Training transform (with data augmentation for robustness)
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    # Geometric augmentation
    transforms.RandomAffine(
        degrees=2,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02),
        shear=2
    ),
    # Color/intensity augmentation
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Validation transform (no augmentation, like inference)
val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class CRNNImagePreprocessor:
    """
    Preprocessing utilities for CRNN model.
    Handles numpy arrays, PIL images, and file paths.
    """
    
    @staticmethod
    def from_numpy(
        image: np.ndarray,
        mode: str = "inference"
    ) -> torch.Tensor:
        """
        Convert numpy array to model input tensor.
        
        Args:
            image: Input as numpy array (grayscale or RGB)
            mode: "inference" (no augmentation) or "training" (with augmentation)
            
        Returns:
            Tensor of shape (1, 1, 64, 512) ready for model
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = Image.fromarray(image, mode='L')
            else:
                image = Image.fromarray(image.astype(np.uint8), mode='RGB')
        
        # Apply appropriate transform
        transform = train_transform if mode == "training" else inference_transform
        tensor = transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    @staticmethod
    def from_pil(
        image: Image.Image,
        mode: str = "inference"
    ) -> torch.Tensor:
        """
        Convert PIL Image to model input tensor.
        
        Args:
            image: PIL Image object
            mode: "inference" or "training"
            
        Returns:
            Tensor of shape (1, 1, 64, 512) ready for model
        """
        transform = train_transform if mode == "training" else inference_transform
        tensor = transform(image)
        return tensor.unsqueeze(0)
    
    @staticmethod
    def from_file(
        image_path: str,
        mode: str = "inference"
    ) -> torch.Tensor:
        """
        Load image from file and convert to model input tensor.
        
        Args:
            image_path: Path to image file
            mode: "inference" or "training"
            
        Returns:
            Tensor of shape (1, 1, 64, 512) ready for model
        """
        image = Image.open(image_path).convert("RGB")
        transform = train_transform if mode == "training" else inference_transform
        tensor = transform(image)
        return tensor.unsqueeze(0)
    
    @staticmethod
    def batch_from_numpy(
        images: list,
        mode: str = "inference"
    ) -> torch.Tensor:
        """
        Convert list of numpy arrays to batch tensor.
        
        Args:
            images: List of numpy arrays
            mode: "inference" or "training"
            
        Returns:
            Tensor of shape (batch_size, 1, 64, 512)
        """
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
    
    Args:
        image: Input image
        target_height: Target height (width scales proportionally)
        
    Returns:
        Resized image
    """
    if len(image.shape) != 2:
        raise ValueError("Expected grayscale image (2D array)")
    
    h, w = image.shape
    scale = target_height / h
    new_w = int(w * scale)
    
    import cv2
    return cv2.resize(image, (new_w, target_height))
