import pandas as pd
import os
from evidently import Report
from evidently.metrics import (
    ValueDrift, DriftedColumnsCount,
    MissingValueCount, DuplicatedRowCount
)
from flask import Flask, Response

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    if not os.path.exists("monitoring/data/current.parquet"):
        return Response("# current.parquet not found yet.\n", mimetype="text/plain")
    
    reference = pd.read_parquet("monitoring/data/reference.parquet")
    reference = reference.rename(columns={'genre_name': 'predicted_genre'})
    reference = reference.drop(columns=['genre'], errors='ignore')
    current = pd.read_parquet("monitoring/data/current.parquet")

    mfcc_features = [f'mfcc_{i}' for i in range(1, 14)]
    metrics = [
        # Data drift metrics - check each MFCC feature for drift
        *[ValueDrift(column=col) for col in mfcc_features],

        # Overall drift summary
        DriftedColumnsCount(),
        
        # Data quality metrics
        *[MissingValueCount(column=col) for col in mfcc_features],
        DuplicatedRowCount(),
    ]
    
    report = Report(metrics=metrics)
    results = report.run(reference_data=reference, current_data=current).dict()

    metrics_output = []
    
    for metric in results['metrics']:
        metric_id = metric['metric_id']
        value = metric['value']

        # Handle ValueDrift
        if metric_id.startswith("ValueDrift(column="):
            column = metric_id.split("(")[1].split("=")[1].rstrip(")")
            metrics_output.append(f'mfcc_value_drift{{feature="{column}"}} {value}')

        # Handle MissingValueCount
        elif metric_id.startswith("MissingValueCount(column="):
            column = metric_id.split("(")[1].split("=")[1].rstrip(")")
            metrics_output.append(f'missing_value_count{{feature="{column}"}} {value["count"]}')

        # Handle DriftedColumnsCount
        elif metric_id.startswith("DriftedColumnsCount("):
            metrics_output.append(f'drifted_columns_count {value["count"]}')
            metrics_output.append(f'drifted_columns_share {value["share"]}')

        # Handle DuplicatedRowCount
        elif metric_id.startswith("DuplicatedRowCount"):
            metrics_output.append(f'duplicated_row_count {value}')

    return Response("\n".join(metrics_output), mimetype="text/plain")
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
