import sys
import os
# Only for Windows
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import yaml
import mlflow
import mlflow.keras
import tensorflow as tf

from classifier.utils.data import prepare_dataset, plot_history
from classifier.models.model import build_model

def train_and_log_model(file_path):
    base_dir = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(base_dir, "..", "config", "mlflow_config.yaml"))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    EXP_NAME    = cfg["experiment_name"]
    MODEL_NAME  = cfg["registered_model_name"]
    params      = cfg["training"]

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXP_NAME)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(file_path)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        model = build_model(X_train.shape[1:], params["l2_reg"], params["learning_rate"])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=params["patience"], restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stop],
            verbose=2
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metrics({
            "test_accuracy": float(test_acc),
            "best_val_accuracy": max(history.history["val_accuracy"])
        })

        fig = plot_history(history)
        mlflow.log_figure(fig, "training_plot.png")

        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print(f"Model registered. Test accuracy: {test_acc:.4f}")

