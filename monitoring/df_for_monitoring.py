import os

import numpy as np
import pandas as pd
import librosa
import requests
from tqdm import tqdm

# Constants
SERVER_URL = "http://127.0.0.1:80/predict"
GENRES_DIR = "../genres"
OUTPUT_PARQUET = "data/monitoring.parquet"
SAMPLE_RATE = 22050
REQUEST_TIMEOUT = 30  # seconds

# Skip known missing files
SKIPPED_FILES = {"jazz.00054.wav", "reggae.00086.wav"}

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

# Storage
records = []

# Walk through genres
for genre in os.listdir(GENRES_DIR):
    genre_path = os.path.join(GENRES_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    for filename in tqdm(os.listdir(genre_path), desc=f"Processing '{genre}'"):
        if not filename.endswith(".wav") or filename in SKIPPED_FILES:
            continue

        file_path = os.path.join(genre_path, filename)

        # Load audio
        try:
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        except (librosa.LibrosaError, OSError, IOError) as e:
            print(f"[WARN] Skipping corrupted file {filename}: {e}")
            continue

        # Extract mean MFCCs (13)
        try:
            mfcc = librosa.feature.mfcc(
                y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
            ).T
            if len(mfcc) < 1:
                print(f"[WARN] Skipping file with empty MFCC: {filename}")
                continue
            mfcc_mean = np.mean(mfcc, axis=0)
        except (ValueError, librosa.LibrosaError) as e:
            print(f"[ERROR] MFCC extraction failed for {filename}: {e}")
            continue

        # Predict using web service
        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "audio/wav")}
                data = {"actual_genre": genre}
                response = requests.post(
                    SERVER_URL, files=files, data=data, timeout=REQUEST_TIMEOUT
                )

            if response.status_code == 200:
                predicted_genre = response.json().get("predicted_genre", "unknown")
            else:
                print(
                    f"[ERROR] API call failed for {filename} with code {response.status_code}"
                )
                predicted_genre = "error"
        except (requests.RequestException, OSError, IOError) as e:
            print(f"[ERROR] API request failed for {filename}: {e}")
            predicted_genre = "error"

        # Create record
        row = {f"mfcc_{i+1}": mfcc_mean[i] for i in range(13)}
        row["actual_genre"] = genre
        row["predicted_genre"] = predicted_genre
        row["file"] = filename
        records.append(row)

# Save to Parquet
df = pd.DataFrame(records)
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"\n✅ Saved monitoring dataset: {OUTPUT_PARQUET} ({len(df)} rows)")
