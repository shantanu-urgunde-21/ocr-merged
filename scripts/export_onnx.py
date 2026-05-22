import os
import sys
from pathlib import Path
import torch

# Force stdout/stderr to use UTF-8 to prevent UnicodeEncodeError on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure parent directory is in path to import local modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.crnn_model import load_crnn_model, IMG_HEIGHT, IMG_WIDTH, VOCAB

def export_to_onnx(
    pytorch_checkpoint_path: str,
    onnx_output_path: str
) -> None:
    """
    Loads PyTorch CRNN model weights and compiles the graph to ONNX format.
    
    Args:
        pytorch_checkpoint_path: Path to the input PyTorch .pth file
        onnx_output_path: Path where the compiled .onnx file will be saved
    """
    print(f"Loading PyTorch checkpoint from: {pytorch_checkpoint_path}")
    if not Path(pytorch_checkpoint_path).exists():
        raise FileNotFoundError(f"Source PyTorch checkpoint not found at {pytorch_checkpoint_path}")

    # 1. Initialize and load model onto CPU (CPU tracing is standard for ONNX exports)
    device = torch.device("cpu")
    vocab_size = len(VOCAB) + 1
    model = load_crnn_model(pytorch_checkpoint_path, device=device, vocab_size=vocab_size)
    model.eval()

    # 2. Create a dummy input tensor matching CRNN height, width, and grayscale channel
    # Shape: (batch_size=1, channels=1, height=64, width=512)
    dummy_input = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH, device=device)

    # 3. Define output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)

    print("Tracing model graph and exporting to ONNX...")
    # 4. Perform dynamic tracing and serialization
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,        # Store trained parameters inside the ONNX file
        opset_version=12,          # Dynamic operators version support
        do_constant_folding=True,  # Run constant folding pre-optimization pass
        input_names=["input"],     # Name of input tensor node
        output_names=["output"],   # Name of output prediction node
        dynamic_axes={
            "input": {0: "batch_size"},   # Allow dynamic batch sizing (e.g. process multiple lines)
            "output": {1: "batch_size"}  # Output matches batch dimension
        }
    )
    print(f"✓ ONNX model compiled and saved successfully to: {onnx_output_path}")

if __name__ == "__main__":
    # Standard paths matching local execution
    DEFAULT_CHECKPOINT = "model_weights/handwriting_recognizer_best.pth"
    DEFAULT_ONNX_OUTPUT = "model_weights/handwriting_recognizer_best.onnx"

    try:
        export_to_onnx(DEFAULT_CHECKPOINT, DEFAULT_ONNX_OUTPUT)
    except Exception as e:
        print(f"❌ Error during ONNX export: {str(e)}")
        print("\nEnsure you have placed handwriting_recognizer_best.pth under model_weights/")
