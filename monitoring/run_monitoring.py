import os
import re
import time

import pandas as pd
from flask import Flask, Response
from evidently import Report, Dataset, DataDefinition, MulticlassClassification
from evidently.metrics import (
    F1Score,
    Accuracy,
    RowCount,
    ValueDrift,
    CategoryCount,
    EmptyRowsCount,
    MissingValueCount,
    DuplicatedRowCount,
    DriftedColumnsCount,
)

app = Flask(__name__)

# File modification time cache to detect changes
_file_cache = {}


def prometheus_safe(name: str) -> str:
    """Sanitize metric names to be Prometheus-compatible"""
    name = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    name = re.sub(r"__+", "_", name)
    name = name.strip("_")
    return name.lower()


def read_parquet_with_cache_bust(file_path):
    """Read parquet file with cache busting to ensure fresh data"""
    try:
        # Get file modification time
        mod_time = os.path.getmtime(file_path)

        # Force a small delay to ensure file writes are complete
        time.sleep(0.1)

        # Always update cache with current modification time
        _file_cache[file_path] = mod_time

        # Read the file fresh each time to avoid pandas caching
        return pd.read_parquet(file_path)

    except (OSError, IOError, FileNotFoundError) as e:
        print(f"Error reading {file_path}: {e}")
        raise
    except pd.errors.ParserError as e:
        print(f"Error parsing parquet file {file_path}: {e}")
        raise


@app.route("/metrics")
def metrics():
    # FIX: Use correct paths that match the volume mount
    reference_path = "monitoring/data/reference.parquet"
    current_path = "monitoring/data/current.parquet"

    if not os.path.exists(reference_path):
        return Response("# reference.parquet not found\n", mimetype="text/plain")
    if not os.path.exists(current_path):
        return Response("# current.parquet not found\n", mimetype="text/plain")

    try:
        # Use cache-busting read to ensure fresh data
        reference_df = read_parquet_with_cache_bust(reference_path)
        current_df = read_parquet_with_cache_bust(current_path)

        # Debug info
        print(f"Reference data shape: {reference_df.shape}")
        print(f"Current data shape: {current_df.shape}")
        print(
            f"Current file last modified: {time.ctime(os.path.getmtime(current_path))}"
        )

    except (OSError, IOError, FileNotFoundError, pd.errors.ParserError) as e:
        error_msg = f"# Error reading parquet files: {str(e)}\n"
        print(error_msg)
        return Response(error_msg, mimetype="text/plain")

    mfcc_features = [f"mfcc_{i}" for i in range(1, 14)]
    categories = sorted(reference_df["predicted_genre"].dropna().unique().tolist())

    multiclass_definition = DataDefinition(
        numerical_columns=mfcc_features,
        classification=[
            MulticlassClassification(
                target="actual_genre", prediction_labels="predicted_genre"
            )
        ],
    )

    try:
        reference_data = Dataset.from_pandas(
            reference_df, data_definition=multiclass_definition
        )
        current_data = Dataset.from_pandas(
            current_df, data_definition=multiclass_definition
        )

        metrics_list = [
            RowCount(),
            EmptyRowsCount(),
            DuplicatedRowCount(),
            *[MissingValueCount(column=col) for col in mfcc_features],
            *[ValueDrift(column=col) for col in mfcc_features],
            DriftedColumnsCount(columns=mfcc_features),
            ValueDrift(column="predicted_genre"),
            CategoryCount(column="predicted_genre", categories=categories),
            Accuracy(),
            F1Score(),
        ]

        report = Report(metrics=metrics_list)
        results = report.run(
            reference_data=reference_data, current_data=current_data
        ).dict()

    except (ValueError, KeyError, AttributeError, TypeError) as e:
        error_msg = f"# Error generating evidently report: {str(e)}\n"
        print(error_msg)
        return Response(error_msg, mimetype="text/plain")

    output_lines = []
    output_lines.append("# HELP evidently_metric Evidently monitoring metric")
    output_lines.append("# TYPE evidently_metric gauge")

    for metric in results["metrics"]:
        raw_name = metric.get("metric_id", "unknown_metric")
        metric_id = prometheus_safe(raw_name)

        value = metric.get("value")

        if isinstance(value, (int, float)):
            output_lines.append(f"{metric_id} {value:.6f}")

        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)):
                    sub_name = prometheus_safe(f"{metric_id}_{sub_key}")
                    output_lines.append(f"{sub_name} {sub_val:.6f}")

        else:
            try:
                numeric_val = float(value)
                output_lines.append(f"{metric_id} {numeric_val:.6f}")
            except (TypeError, ValueError):
                pass

    return Response("\n".join(output_lines) + "\n", mimetype="text/plain")


@app.route("/health")
def health():
    """Health check endpoint"""
    reference_path = "monitoring/data/reference.parquet"
    current_path = "monitoring/data/current.parquet"

    status = {
        "reference_exists": os.path.exists(reference_path),
        "current_exists": os.path.exists(current_path),
        "timestamp": time.time(),
    }

    if os.path.exists(current_path):
        status["current_size"] = os.path.getsize(current_path)
        status["current_modified"] = os.path.getmtime(current_path)

    return status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
