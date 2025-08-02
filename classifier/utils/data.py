import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Use a backend without graphical interface (avoid errors with Prefect/tasks)
matplotlib.use("Agg")


def load_data(data_path):
    data = np.load(data_path)
    X = data["mfcc"][..., np.newaxis]
    y = data["labels"]
    print(f"Loaded X:{X.shape}, y:{y.shape}")
    return X, y


def prepare_dataset(data_path, test_size=0.2, val_size=0.25):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_history(history, save_path="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_title("Model Loss")
    ax2.legend()

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to: {save_path}")

    return fig
