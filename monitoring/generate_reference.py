#!/usr/bin/env python3
"""
monitoring/generate_reference.py

Generates a reference dataset (reference.parquet) from data_10.npz.
Each sample includes the mean of each MFCC coefficient (13 values) and its corresponding genre label.
"""

import os
import numpy as np
import pandas as pd
import argparse

def generate_reference(npz_path: str, output_path: str):
    # 1) Load the .npz dataset
    data = np.load(npz_path, allow_pickle=True)
    mfccs = data["mfcc"]        # shape: (n_samples, time_steps, 13)
    labels = data["labels"]     # shape: (n_samples,)
    mapping = data.get("mapping", None)  # optional: list of genre names

    n_samples = mfccs.shape[0]
    print(f"[INFO] Loaded {n_samples} samples from '{npz_path}'")

    # 2) Compute mean MFCCs for each sample (13 values per sample)
    mfcc_means = [np.mean(mfccs[i], axis=0) for i in range(n_samples)]
    mfcc_means = np.stack(mfcc_means, axis=0)  # shape: (n_samples, 13)

    # 3) Build a DataFrame with MFCC means and genre labels
    col_names = [f"mfcc_{j+1}" for j in range(mfcc_means.shape[1])]
    df = pd.DataFrame(mfcc_means, columns=col_names)
    df["genre"] = labels

    # Optional: add genre name column if mapping is available
    if mapping is not None:
        df["genre_name"] = df["genre"].apply(lambda idx: mapping[int(idx)])

    # 4) Save as a Parquet file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[SUCCESS] Reference dataset saved to '{output_path}' "
          f"({df.shape[0]} rows, {df.shape[1]} columns)")

def main():
    parser = argparse.ArgumentParser(
        description="Generate reference.parquet from a compressed MFCC .npz file"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data_10.npz",
        help="Path to the input .npz file (default: data_10.npz)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="monitoring/data/reference.parquet",
        help="Path to the output Parquet file (default: monitoring/data/reference.parquet)"
    )
    args = parser.parse_args()
    generate_reference(args.input, args.output)

if __name__ == "__main__":
    main()
