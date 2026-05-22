"""OCR Models Package"""

try:
    import torch
    from .crnn_model import CRNN, load_crnn_model, ctc_decode, predict_text
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    CRNN = None
    load_crnn_model = None
    ctc_decode = None
    predict_text = None

# Pure NumPy and general components
from .constants import VOCAB, CTC_BLANK, char_to_int, int_to_char, IMG_HEIGHT, IMG_WIDTH
from .tesseract_model import TesseractModel
from .model_registry import ModelRegistry
from .onnx_model import ONNXCRNNModel

__all__ = [
    "CRNN",
    "load_crnn_model",
    "ctc_decode",
    "predict_text",
    "TesseractModel",
    "ModelRegistry",
    "ONNXCRNNModel",
    "VOCAB",
    "CTC_BLANK",
    "char_to_int",
    "int_to_char",
    "IMG_HEIGHT",
    "IMG_WIDTH",
]

