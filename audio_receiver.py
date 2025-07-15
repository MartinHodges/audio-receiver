import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments

import os
from flask import Flask, request, jsonify
import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
import infer
from io import BytesIO

app = Flask(__name__)

TARGET_SAMPLE_RATE = 16000  # PANNs target sample rate
SAMPLE_RATE = 48000  # Sample rate from the audio processing
UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    current_time = datetime.now()

    # Format the datetime into a string (e.g., 20250709_215958)
    # You can customize this format using strftime directives.
    # %Y: Year with century (e.g., 2025)
    # %m: Month as a zero-padded decimal number (01-12)
    # %d: Day of the month as a zero-padded decimal number (01-31)
    # %H: Hour (24-hour clock) as a zero-padded decimal number (00-23)
    # %M: Minute as a zero-padded decimal number (00-59)
    # %S: Second as a zero-padded decimal number (00-59)
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename.replace(".wav", f"-{datetime_str}.wav"))
    # file.save(filepath) # comment this out if the file does not need to be saved    

    try:
        sr, audio = wavfile.read(BytesIO(file.read()))

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz for PANNs
        target_sr = TARGET_SAMPLE_RATE
        if sr != target_sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        if len(audio) != 3 * TARGET_SAMPLE_RATE:
          return jsonify({'error': f'Expected {3 * SAMPLE_RATE} samples (3 second window), got {len(audio)}'}), 400

        # Normalize to [-1, 1] if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        infer.infer_against_raw(audio)

        return jsonify({'message': 'File processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3030)