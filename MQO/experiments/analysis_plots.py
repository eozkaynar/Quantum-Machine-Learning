import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import os

def plot_train_curve(train_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "train_loss_curve.png"))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.show()

def print_classification_report(y_true, y_pred):
    report = metrics.classification_report(y_true, y_pred, digits=4)
    print("Classification Report:\n")
    print(report)

def main(loss_path="output/classical/train_losses.npy",
         pred_path="output/classical/test_predictions.csv",
         save_dir="output/classical/"):

    # Load loss
    train_losses = np.load(loss_path)

    # Load predictions
    df = pd.read_csv(pred_path)
    y_true = df["true_value"].to_numpy()
    y_pred = df["prediction"].to_numpy()

    # Plot loss curve
    plot_train_curve(train_losses, save_path=save_dir)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=save_dir)

    # Metrics
    print_classification_report(y_true, y_pred)

if __name__ == "__main__":
    main()
