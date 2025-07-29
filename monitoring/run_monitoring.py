import os
import re
import pandas as pd
from evidently import MulticlassClassification, Dataset, DataDefinition, Report
from evidently.metrics import (
    RowCount,
    EmptyRowsCount,
    DuplicatedRowCount,
    MissingValueCount,
    ValueDrift,
    DriftedColumnsCount,
    CategoryCount,
    Accuracy,
    F1Score
)
from flask import Flask, Response

app = Flask(__name__)

# Sanitize metric names to be Prometheus-compatible
def prometheus_safe(name: str) -> str:
    # Replace any illegal character with underscore
    name = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    # Remove consecutive underscores
    name = re.sub(r"__+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name.lower()

@app.route("/metrics")
def metrics():
    reference_path = "data/reference.parquet"
    current_path = "data/current.parquet"

    if not os.path.exists(reference_path):
        return Response("# reference.parquet not found\n", mimetype="text/plain")
    if not os.path.exists(current_path):
        return Response("# current.parquet not found\n", mimetype="text/plain")

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    mfcc_features = [f"mfcc_{i}" for i in range(1, 14)]
    categories = sorted(reference_df["predicted_genre"].dropna().unique().tolist())

    multiclass_definition = DataDefinition(
        numerical_columns=mfcc_features,
        classification=[
            MulticlassClassification(
                target="actual_genre",
                prediction_labels="predicted_genre"
            )
        ],
    )

    reference_data = Dataset.from_pandas(reference_df, data_definition=multiclass_definition)
    current_data = Dataset.from_pandas(current_df, data_definition=multiclass_definition)

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
        F1Score()
    ]

    report = Report(metrics=metrics_list)
    results = report.run(reference_data=reference_data, current_data=current_data).dict()

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
