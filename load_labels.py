import csv
import torch.nn.functional as F # For sigmoid activation

# --- Assuming you have the path to your audioset_tagging_cnn repository ---
# You might need to adjust this path based on where you cloned the repo
REPO_ROOT = '../audioset_tagging_cnn' # Or wherever your repo root is
LABELS_FILE_PATH = f'{REPO_ROOT}/metadata/class_labels_indices.csv'

# Load the class labels
def load_labels(filepath = LABELS_FILE_PATH):
    """Loads the AudioSet class labels from a CSV file."""
    labels = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header row
        for row in reader:
            # Assuming format: index, mid, display_name
            labels.append(row[2]) # Get the display_name
    return labels
