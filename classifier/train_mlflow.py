# train_mlflow.py
import os, yaml, numpy as np
import mlflow, mlflow.keras
import tensorflow as tf

from src.utils import prepare_dataset, plot_history
from src.model import build_model

# === Load config ===
with open("mlflow_config.yaml") as f:
    cfg = yaml.safe_load(f)

EXP_NAME    = cfg["experiment_name"]
MODEL_NAME  = cfg["registered_model_name"]
params      = cfg["training"]
paths       = cfg["paths"]

DATA_PATH   = paths["data"]
OUTPUT_DIR  = paths["output_dir"]
OUTPUT_FILE = os.path.join(OUTPUT_DIR, paths["model_name"])
os.makedirs(OUTPUT_DIR, exist_ok=True)



def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXP_NAME)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(DATA_PATH)

    with mlflow.start_run() as run:
        # Log training parameters
        mlflow.log_params(params)

        # Build & train model
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

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        best_val_acc = max(history.history["val_accuracy"])

        # Log metrics
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("best_val_accuracy", float(best_val_acc))

        # Save and log figure
        fig = plot_history(history)
        mlflow.log_figure(fig, "training_plot.png")

        # Register model
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print(f"✅ Model registered. Test accuracy: {test_acc:.4f}")
        print(f"🔗 Run: {mlflow.get_run(run.info.run_id).info.run_id}")



if __name__ == "__main__":
    main()
