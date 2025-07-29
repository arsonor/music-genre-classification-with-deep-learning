import pandas as pd
import os
import re
from evidently import MulticlassClassification, Dataset, DataDefinition, Report
from evidently.metrics import *
from flask import Flask, Response

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    # Ensure both reference and current exist
    if not os.path.exists("data/reference.parquet"):
        return Response("# reference.parquet not found.\n", mimetype="text/plain")
    if not os.path.exists("data/current.parquet"):
        return Response("# current.parquet not found.\n", mimetype="text/plain")
    
    reference_df = pd.read_parquet("data/reference.parquet")
    current_df = pd.read_parquet("data/current.parquet")

    mfcc_features = [f"mfcc_{i}" for i in range(1, 14)]
    categories = sorted(reference_df["predicted_genre"].dropna().unique().tolist())

    multiclass_definition = DataDefinition(
        numerical_columns=mfcc_features,
        classification=[MulticlassClassification(
            target="actual_genre",
            prediction_labels="predicted_genre"
        )],
    )

    reference_data = Dataset.from_pandas(reference_df, data_definition=multiclass_definition)
    current_data = Dataset.from_pandas(current_df, data_definition=multiclass_definition)

    metrics = [
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
    
    report = Report(metrics=metrics)
    results = report.run(reference_data=reference_data, current_data=current_data).dict()

    metrics_output = []
    metrics_output.append("# HELP evidently_metric Evidently monitoring metric")
    metrics_output.append("# TYPE evidently_metric gauge")

    for metric in results["metrics"]:
        metric_id = metric.get("metric_id").lower().replace("()", "").replace(" ", "_")
        value = metric.get("value")

        # Handle numeric values
        if isinstance(value, (int, float)):
            metrics_output.append(f"{metric_id} {value:.6f}")

        # Handle dict values (like drift counts)
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)):
                    sub_name = f"{metric_id}_{sub_key}".replace(" ", "_").replace(",", "_")
                    metrics_output.append(f"{sub_name} {sub_val:.6f}")

        # Handle numpy floats
        else:
            try:
                metrics_output.append(f"{metric_id} {float(value):.6f}")
            except:
                pass

    return Response("\n".join(metrics_output), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
