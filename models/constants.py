"""
OCR Vocabulary and Dimension Constants
Kept PyTorch-free to allow lightweight ONNX deployments.
"""

# Vocabulary from training data
VOCAB = """ !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
CTC_BLANK = 0

# Character mappings
char_to_int = {char: i + 1 for i, char in enumerate(VOCAB)}
int_to_char = {i + 1: char for i, char in enumerate(VOCAB)}

# Image dimensions (must match training)
IMG_HEIGHT = 64
IMG_WIDTH = 512
