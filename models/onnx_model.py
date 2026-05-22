import onnxruntime as ort
import numpy as np
from .constants import VOCAB, CTC_BLANK, int_to_char


def ctc_decode_numpy(log_probs: np.ndarray) -> list:
    """
    Decode CTC output (numpy array) to readable text.
    
    Args:
        log_probs: Log probabilities from model, shape (sequence_length, batch_size, num_chars)
        
    Returns:
        List of decoded text strings (one per batch item)
    """
    # Get argmax predictions along the character probability axis (axis 2)
    # Transpose to get shape (batch_size, sequence_length)
    preds = log_probs.argmax(axis=2).T
    
    decoded_texts = []
    for pred in preds:
        # 1. Collapse consecutive identical tokens first (including blanks)
        collapsed = []
        prev = None
        for val in pred:
            val_item = int(val)
            if val_item != prev:
                collapsed.append(val_item)
                prev = val_item
                
        # 2. Filter out the CTC blank tokens and map to characters
        chars = [int_to_char.get(c, '') for c in collapsed if c != CTC_BLANK]
        decoded_texts.append(''.join(chars))
        
    return decoded_texts


class ONNXCRNNModel:
    """
    ONNX Runtime adapter for CRNN handwritten text recognition.
    Integrates seamlessly with preprocessors and CTC decoding without PyTorch dependencies.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX session.
        
        Args:
            model_path: Path to the compiled .onnx file
        """
        self.model_path = model_path
        # Load ONNX session with default CPU/GPU execution providers
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, image_input) -> str:
        """
        Run fast ONNX inference on a preprocessed image (tensor or numpy array).
        
        Args:
            image_input: PyTorch tensor or NumPy array of shape (1, 1, 64, 512)
            
        Returns:
            Decoded handwriting transcription string
        """
        # Convert PyTorch tensor to NumPy array if needed
        if hasattr(image_input, "cpu"):
            img_numpy = image_input.cpu().numpy()
        else:
            img_numpy = np.asarray(image_input, dtype=np.float32)
        
        # Run highly optimized inference graph in C++
        outputs = self.session.run(None, {self.input_name: img_numpy})
        log_probs = outputs[0]  # Shape: (sequence_length, batch_size, num_chars)
        
        # CTC decode using pure NumPy decoding
        decoded = ctc_decode_numpy(log_probs)
        return decoded[0] if decoded else ""
