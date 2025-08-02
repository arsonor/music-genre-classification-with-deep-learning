import os
import random

# from datetime import datetime, timezone
import pandas as pd
from flask import Flask, jsonify, request
from genre_prediction_service import Genre_Prediction_Service

# Instantiate Flask app
app = Flask(__name__)

# Path to the monitoring output
CURRENT_PARQUET_PATH = "monitoring/data/current.parquet"


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to predict genre and log input features to current.parquet

    Returns:
        json: { "genre": "blues" }
    """

    # Get file from POST request and save temporarily
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000)) + ".wav"
    audio_file.save(file_name)

    # Extract optional actual label
    actual_genre = request.form.get("actual_genre", default=None)

    # Instantiate genre prediction service singleton
    gps = Genre_Prediction_Service()

    # Predict genre
    predicted_genre = gps.predict(file_name)

    # Extract MFCC mean vector (13 features) for logging
    mfcc_vector = gps.extract_mean_mfcc(file_name)

    # Delete temp audio file
    os.remove(file_name)

    # Build a row DataFrame with MFCCs + genre + timestamp
    col_names = [f"mfcc_{i+1}" for i in range(13)]
    df_row = pd.DataFrame([mfcc_vector], columns=col_names)
    df_row["predicted_genre"] = predicted_genre
    df_row["actual_genre"] = actual_genre
    # df_row["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Append to current.parquet (create if not exists)
    os.makedirs(os.path.dirname(CURRENT_PARQUET_PATH), exist_ok=True)
    if os.path.exists(CURRENT_PARQUET_PATH):
        df_existing = pd.read_parquet(CURRENT_PARQUET_PATH)
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_combined = df_row

    df_combined.to_parquet(CURRENT_PARQUET_PATH, index=False)

    # Return prediction result
    return jsonify({"predicted_genre": predicted_genre})


if __name__ == "__main__":
    print("Starting genre prediction service...")
    app.run(debug=False)
