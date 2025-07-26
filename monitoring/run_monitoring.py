import pandas as pd
from evidently import Report
from evidently.metrics import (
    ValueDrift,
    MissingValueCount,
)
from flask import Flask, Response

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    reference = pd.read_parquet("monitoring/data/reference.parquet")
    current = pd.read_parquet("monitoring/data/current.parquet")

    report = Report(metrics=[
    *[ValueDrift(column=f"mfcc_{i+1}") for i in range(13)],
    *[MissingValueCount(column=f"mfcc_{i+1}") for i in range(13)],
    ])

    results = report.run(reference_data=reference, current_data=current).dict()

    metrics_output = []
    
    # First 13 metrics: ValueDrift
    for i in range(13):
        drift_score = results['metrics'][i]['value']
        metrics_output.append(f'mfcc_value_drift{{feature="mfcc_{i+1}"}} {drift_score:.6f}')

    # Next 13 metrics: MissingValueCount
    for i in range(13, 26):
        feature_index = i - 13 + 1
        missing_count = results['metrics'][i]['value']['count']
        metrics_output.append(f'missing_value_count{{feature="mfcc_{feature_index}"}} {missing_count}')

    return Response("\n".join(metrics_output), mimetype="text/plain")
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
