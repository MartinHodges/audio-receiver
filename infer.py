import torch
import numpy as np
import sys
import os

panns_repo_path = '../audioset_tagging_cnn/pytorch' 

if os.path.exists(panns_repo_path) and panns_repo_path not in sys.path:
    sys.path.append(panns_repo_path)
    print(f"Added {panns_repo_path} to sys.path.")
else:
    print(f"Warning: {panns_repo_path} not found or already in sys.path. "
          "Please ensure the PANNs repository is cloned correctly.")

from load_labels import load_labels
from models import Cnn14

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define PANNs model parameters for Cnn14_mAP=0.438.pth (16kHz)
target_panns_sample_rate = 16000
panns_window_size = 512
panns_hop_size = 160
panns_mel_bins = 64
panns_fmin = 50
panns_fmax = 8000
panns_classes_num_audioset = 527

# Path to your downloaded weights
PANN_BASE_WEIGHTS = './pretrained/Cnn14_16k_mAP=0.438.pth'

# Check if weights file exists
if not os.path.exists(PANN_BASE_WEIGHTS):
    print(f"Error: PANNs weights not found at {PANN_BASE_WEIGHTS}. Please download them.")
    sys.exit(1)

print("Instantiating Cnn14 model...")
try:
    model = Cnn14(sample_rate=target_panns_sample_rate,
                  window_size=panns_window_size,
                  hop_size=panns_hop_size,
                  mel_bins=panns_mel_bins,
                  fmin=panns_fmin,
                  fmax=panns_fmax,
                  classes_num=panns_classes_num_audioset)
    print("Cnn14 model instantiated successfully.")
except Exception as e:
    print(f"Error instantiating Cnn14: {e}")
    sys.exit(1)

print(f"Loading weights from {PANN_BASE_WEIGHTS}...")
try:
    checkpoint = torch.load(PANN_BASE_WEIGHTS, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")
    sys.exit(1)

model.to(device)
model.eval() # Set to eval mode

def infer_against_mel_spectrogram(filename):
    print("Not yet implemented (and not recommended with PANN models)")

def infer_against_raw(audio):
    global panns_repo_path

    # Convert to tensor and add batch dimension
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # shape: (1, num_samples)

    # Run inference
    model.to(device)
    model.eval()
    with torch.no_grad():
        model_output = model(waveform.to(device))
        clipwise_output = model_output["clipwise_output"]
        probabilities = torch.sigmoid(clipwise_output)
        probabilities_np = probabilities.cpu().numpy()

    # ANSI escape codes
    CURSOR_UP_ONE = "\033[1A"
    CLEAR_LINE = "\033[K" # Clears from cursor to the end of the line

    # Display top predictions for your clip
    audioset_labels = load_labels()
    clip_probabilities = probabilities_np[0]
    sorted_indices = clip_probabilities.argsort()[::-1]
    print("Top 3 predicted labels and their probabilities:")    
    for i in range(3):
        label_index = sorted_indices[i]
        label_name = audioset_labels[label_index]
        probability = clip_probabilities[label_index]
        sys.stdout.write(CLEAR_LINE)
        sys.stdout.write(f"{i+1}. {label_name}: {probability:.4f}\n")
    
    for _ in range(5):
        sys.stdout.write(CURSOR_UP_ONE)
    sys.stdout.flush()