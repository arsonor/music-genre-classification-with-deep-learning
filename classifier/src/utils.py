import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(data_path):
    data = np.load(data_path)
    X = data["mfcc"][..., np.newaxis]
    y = data["labels"]
    print(f"Loaded X:{X.shape}, y:{y.shape}")
    return X, y

def prepare_dataset(data_path, test_size=0.2, val_size=0.25):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val,  y_train, y_val  = train_test_split(
        X_train, y_train, test_size=val_size
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))
    ax1.plot(history.history["accuracy"], label="train acc")
    ax1.plot(history.history["val_accuracy"], label="val acc")
    ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.set_title("Accuracy")
    ax2.plot(history.history["loss"], label="train loss")
    ax2.plot(history.history["val_loss"], label="val loss")
    ax2.set_ylabel("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.set_title("Loss")
    fig.tight_layout()
    return fig
