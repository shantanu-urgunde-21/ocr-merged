"""
CRNN (Convolutional Recurrent Neural Network) Model for Handwritten Text Recognition
Trained on IAM Handwriting Database
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional
from pathlib import Path


from .constants import VOCAB, CTC_BLANK, char_to_int, int_to_char, IMG_HEIGHT, IMG_WIDTH


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for text recognition.
    
    Architecture:
    - CNN: Extracts visual features from images
    - RNN (Bi-LSTM): Analyzes sequence of features, captures context
    - FC: Output layer for character prediction
    - CTC Loss: Enables training on unaligned text sequences
    """
    
    def __init__(self, num_chars: int):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        
        # Convolutional layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Map CNN output to sequence features
        self.map_to_seq = nn.Linear(64 * (IMG_HEIGHT // 4), 64)
        
        # Bidirectional LSTM for sequence modeling
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.25
        )
        
        # Fully connected layer for character prediction
        self.fc = nn.Linear(256, num_chars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            log_probs: Log probabilities of shape (sequence_length, batch_size, num_chars)
        """
        # CNN feature extraction
        x = self.cnn(x)
        
        # Reshape for RNN input: (batch, height, width, channels) -> (batch, width, features)
        x = x.permute(0, 3, 1, 2)
        b, w, c, h = x.size()
        x = x.view(b, w, c * h)
        
        # Map to sequence dimension
        x = self.map_to_seq(x)
        
        # Permute for RNN: (batch, sequence, features) -> (sequence, batch, features)
        x = x.permute(1, 0, 2)
        
        # LSTM processing
        x, _ = self.rnn(x)
        
        # Character prediction
        x = self.fc(x)
        
        # Log softmax for CTC loss compatibility
        x = nn.functional.log_softmax(x, dim=2)
        
        return x


def ctc_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Decode CTC output to readable text.
    
    Args:
        log_probs: Log probabilities from model, shape (sequence_length, batch_size, num_chars)
        
    Returns:
        List of decoded text strings (one per batch item)
    """
    # Get argmax predictions
    preds = log_probs.argmax(dim=2).permute(1, 0)  # (batch_size, sequence_length)
    
    decoded_texts = []
    for pred in preds:
        # 1. Collapse consecutive identical tokens first (including blanks)
        collapsed = []
        prev = None
        for val in pred:
            val_item = val.item()
            if val_item != prev:
                collapsed.append(val_item)
                prev = val_item
                
        # 2. Filter out the CTC blank tokens and map to characters
        chars = [int_to_char.get(c, '') for c in collapsed if c != CTC_BLANK]
        decoded_texts.append(''.join(chars))
        
    return decoded_texts


def load_crnn_model(
    model_path: str,
    device: Optional[torch.device] = None,
    vocab_size: Optional[int] = None
) -> CRNN:
    """
    Load a trained CRNN model from disk.
    
    Args:
        model_path: Path to .pth checkpoint file
        device: Device to load model on (cuda/cpu). Auto-detected if None.
        vocab_size: Number of characters (default: len(VOCAB) + 1)
        
    Returns:
        Loaded CRNN model in eval mode
        
    Raises:
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If model loading fails
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if vocab_size is None:
        vocab_size = len(VOCAB) + 1
    
    # Create model architecture
    model = CRNN(num_chars=vocab_size).to(device)
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    return model


def predict_text(
    model: CRNN,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None
) -> str:
    """
    Run inference on a single image and decode output.
    
    Args:
        model: Loaded CRNN model
        image_tensor: Preprocessed image tensor of shape (1, 1, height, width)
        device: Device to run inference on
        
    Returns:
        Decoded text string
    """
    if device is None:
        device = next(model.parameters()).device
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        log_probs = model(image_tensor)
    
    decoded = ctc_decode(log_probs)
    return decoded[0] if decoded else ""
