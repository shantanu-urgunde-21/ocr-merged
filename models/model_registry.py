"""
Model Registry and Factory Pattern
Manages loading and selection of different OCR models
"""

from typing import Union, Optional, Literal
import torch

from .crnn_model import CRNN, load_crnn_model
from .tesseract_model import TesseractModel


ModelType = Literal["crnn", "tesseract"]


class ModelRegistry:
    """
    Factory for creating and managing OCR model instances.
    Supports multiple models with lazy loading to optimize memory.
    """
    
    # Cache for loaded models
    _models = {}
    _devices = {}
    
    @classmethod
    def get_model(
        cls,
        model_type: ModelType,
        device: Optional[torch.device] = None,
        model_path: Optional[str] = None,
        **kwargs
    ) -> Union[CRNN, TesseractModel]:
        """
        Get a model instance, creating it if needed.
        
        Args:
            model_type: "crnn" or "tesseract"
            device: Torch device (cuda/cpu). Auto-detected if None.
            model_path: Path to CRNN .pth file (required for CRNN, ignored for Tesseract)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model_type is unknown or required args missing
            FileNotFoundError: If CRNN model_path doesn't exist
        """
        if model_type not in ["crnn", "tesseract"]:
            raise ValueError(f"Unknown model type: {model_type}. Use 'crnn' or 'tesseract'.")
        
        # Auto-detect device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Return cached model if available and device matches
        if model_type in cls._models and cls._devices.get(model_type) == device:
            return cls._models[model_type]
        
        # Create new model
        if model_type == "crnn":
            if model_path is None:
                raise ValueError("model_path is required for CRNN model")
            
            model = load_crnn_model(model_path, device=device)
            cls._models["crnn"] = model
            cls._devices["crnn"] = device
            return model
        
        elif model_type == "tesseract":
            config = kwargs.get("config", "--psm 7")
            model = TesseractModel(config=config)
            cls._models["tesseract"] = model
            cls._devices["tesseract"] = device
            return model
    
    @classmethod
    def get_available_models(cls) -> list:
        """Return list of available model types."""
        return ["crnn", "tesseract"]
    
    @classmethod
    def clear_cache(cls, model_type: Optional[ModelType] = None) -> None:
        """
        Clear cached models to free memory.
        
        Args:
            model_type: Clear specific model. If None, clear all.
        """
        if model_type is None:
            cls._models.clear()
            cls._devices.clear()
        else:
            cls._models.pop(model_type, None)
            cls._devices.pop(model_type, None)
    
    @classmethod
    def get_model_info(cls, model_type: ModelType) -> dict:
        """
        Get information about a model type.
        
        Args:
            model_type: Model type to query
            
        Returns:
            Dictionary with model info
        """
        info = {
            "crnn": {
                "name": "CRNN (Convolutional Recurrent Neural Network)",
                "best_for": "Handwritten text recognition",
                "languages": "English (trained on IAM dataset)",
                "vocab_size": 80,
                "input_size": (64, 512),
                "requires_file": True,
                "accuracy_notes": "Trained on clean handwritten lines, best accuracy with preprocessing",
            },
            "tesseract": {
                "name": "Tesseract OCR",
                "best_for": "General printed text (fallback option)",
                "languages": "Multiple languages supported",
                "vocab_size": "Unlimited",
                "input_size": "Variable",
                "requires_file": False,
                "accuracy_notes": "Works on general text but less accurate on handwriting",
            },
        }
        
        if model_type not in info:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return info[model_type]
