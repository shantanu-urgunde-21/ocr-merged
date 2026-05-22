"""
Model Registry and Factory Pattern
Manages loading and selection of different OCR models
"""

from typing import Union, Optional, Literal, Any

from .tesseract_model import TesseractModel
from .onnx_model import ONNXCRNNModel


ModelType = Literal["crnn", "tesseract", "crnn_onnx"]


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
        device: Optional[Any] = None,
        model_path: Optional[str] = None,
        **kwargs
    ) -> Union[Any, TesseractModel, ONNXCRNNModel]:
        """
        Get a model instance, creating it if needed.
        
        Args:
            model_type: "crnn", "tesseract", or "crnn_onnx"
            device: Torch device (cuda/cpu) or custom device identifier. Auto-detected if None.
            model_path: Path to CRNN file (required for CRNN / ONNX, ignored for Tesseract)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model_type is unknown or required args missing
            FileNotFoundError: If CRNN model_path doesn't exist
        """
        if model_type not in ["crnn", "tesseract", "crnn_onnx"]:
            raise ValueError(f"Unknown model type: {model_type}. Use 'crnn', 'tesseract', or 'crnn_onnx'.")
        
        # Auto-detect device
        if device is None:
            try:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                device = "cpu"
        
        # Return cached model if available and device matches
        if model_type in cls._models and cls._devices.get(model_type) == device:
            return cls._models[model_type]
        
        # Create new model
        if model_type == "crnn":
            if model_path is None:
                raise ValueError("model_path is required for CRNN model")
            
            try:
                import torch
                from .crnn_model import load_crnn_model
            except ImportError:
                raise RuntimeError(
                    "PyTorch (torch/torchvision) is required for PyTorch CRNN model inference. "
                    "Please install PyTorch or select 'crnn_onnx' for fast CPU inference using ONNX Runtime."
                )
            
            model = load_crnn_model(model_path, device=device)
            cls._models["crnn"] = model
            cls._devices["crnn"] = device
            return model
        
        elif model_type == "crnn_onnx":
            if model_path is None:
                raise ValueError("model_path is required for CRNN ONNX model")
            
            # Auto-resolve .pth extension to .onnx if passed
            if model_path.endswith(".pth"):
                model_path = model_path.replace(".pth", ".onnx")
                
            model = ONNXCRNNModel(model_path)
            cls._models["crnn_onnx"] = model
            cls._devices["crnn_onnx"] = device
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
        return ["crnn", "tesseract", "crnn_onnx"]
    
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
            "crnn_onnx": {
                "name": "CRNN (ONNX Compiled Version)",
                "best_for": "Ultra-fast production handwritten text recognition on CPU",
                "languages": "English (trained on IAM dataset)",
                "vocab_size": 80,
                "input_size": (64, 512),
                "requires_file": True,
                "accuracy_notes": "C++ optimized graph execution. Speeds up CPU inference up to 5x.",
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
