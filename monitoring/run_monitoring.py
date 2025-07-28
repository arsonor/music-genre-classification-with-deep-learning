import pandas as pd
import os
from evidently import MulticlassClassification
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.metrics import *
from flask import Flask, Response

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    if not os.path.exists("data/current.parquet"):
        return Response("# current.parquet not found yet.\n", mimetype="text/plain")
    
    reference_df = pd.read_parquet("data/reference.parquet")
    current_df = pd.read_parquet("data/current.parquet")

    mfcc_features = [f'mfcc_{i}' for i in range(1, 14)]

    multiclass_definition=DataDefinition(
        numerical_columns=mfcc_features,
        classification=[MulticlassClassification(
            target="actual_genre",
            prediction_labels="predicted_genre"
            )
        ],
    )

    multiclass_reference_data = Dataset.from_pandas(
    reference_df,
    data_definition=multiclass_definition,
    )

    multiclass_current_data = Dataset.from_pandas(
    current_df,
    data_definition=multiclass_definition,
    )

    metrics = [
    # Data quality metrics
    RowCount(),
    EmptyRowsCount(),
    DuplicatedRowCount(),
    *[MissingValueCount(column=col) for col in mfcc_features],

    # Data drift metrics - check each MFCC feature for drift
    *[ValueDrift(column=col) for col in mfcc_features],
    # Overall drift summary
    DriftedColumnsCount(columns=mfcc_features),

    # Prediction drift (on predicted genre)
    ValueDrift(column="predicted_genre"),
    CategoryCount(
    column="predicted_genre",
    categories=sorted(reference_df["predicted_genre"].unique().tolist())
    ),

    # Classification performance
    Accuracy(),
    F1Score()
    ]
    
    report = Report(metrics=metrics)
    results = report.run(
        reference_data=multiclass_reference_data,
        current_data=multiclass_current_data
        ).dict()

    metrics_output = []
    
    

    return Response("\n".join(metrics_output), mimetype="text/plain")
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
