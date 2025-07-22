import os
import numpy as np
from prefect import task

@task
def download_and_validate_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "data_10.npz")
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    print(f"Data found at: {file_path}")
    return file_path

@task
def extract_features(file_path):
    data = np.load(file_path)
    X = data["mfcc"]
    y = data["labels"]
    print(f"Extracted: X {X.shape}, y {y.shape}")
    return X, y
