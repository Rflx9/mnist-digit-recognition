"""
evaluate.py — Model Evaluation Script
=======================================
Loads the trained CNN model and evaluates it on the MNIST test set.
Produces:
  - Overall accuracy score
  - Per-class classification report
  - Confusion matrix heatmap

Usage:
    python evaluate.py
    python evaluate.py --model path/to/model.keras
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DEFAULT_MODEL_PATH = "mnist_cnn_model.keras"
PLOTS_DIR          = "plots"
CLASS_NAMES        = [str(i) for i in range(10)]   # ["0", "1", ..., "9"]


def load_test_data():
    """
    Load and preprocess the MNIST test set.

    Returns:
        x_test (np.ndarray): Normalized images of shape (10000, 28, 28, 1)
        y_test (np.ndarray): Integer labels of shape (10000,)
    """
    print("📦 Loading MNIST test data...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = x_test.astype("float32") / 255.0          # Normalize to [0, 1]
    x_test = np.expand_dims(x_test, axis=-1)            # Add channel dim

    print(f"   Test samples  : {len(x_test):,}")
    print(f"   Image shape   : {x_test.shape[1:]}\n")
    return x_test, y_test


def load_model(model_path):
    """
    Load a saved Keras model from disk.

    Args:
        model_path (str): Path to the .keras or .h5 model file.

    Returns:
        Loaded Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model not found at '{model_path}'\n"
            f"   Run 'python train.py' first to train and save the model."
        )

    print(f"🔄 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!\n")
    return model


def evaluate_model(model, x_test, y_test):
    """
    Run inference on the test set and return predictions.

    Args:
        model: Trained Keras model.
        x_test: Test images.
        y_test: True labels.

    Returns:
        y_pred (np.ndarray): Predicted class indices.
        y_proba (np.ndarray): Softmax probabilities for each class.
    """
    print("🔍 Running predictions on test set...")

    # Get probability distributions over 10 classes for each image
    y_proba = model.predict(x_test, verbose=1)

    # Convert probabilities → class predictions (take argmax)
    y_pred  = np.argmax(y_proba, axis=1)

    return y_pred, y_proba


def print_evaluation_report(y_test, y_pred):
    """
    Print accuracy and per-class precision/recall/F1 report.
    """
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("             📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Overall Test Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print("\n" + "-" * 60)
    print("  Per-Class Classification Report:")
    print("-" * 60)

    report = classification_report(
        y_test, y_pred,
        target_names=[f"Digit {c}" for c in CLASS_NAMES],
        digits=4
    )
    print(report)
    print("=" * 60 + "\n")

    return acc


def plot_confusion_matrix(y_test, y_pred):
    """
    Generate and save a styled confusion matrix heatmap.

    A confusion matrix shows where the model gets confused:
      - Rows    → True labels
      - Columns → Predicted labels
      - Diagonal cells → Correct predictions (want these high)
      - Off-diagonal   → Misclassifications (want these low/zero)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Compute the confusion matrix (10×10 for 10 digit classes)
    cm = confusion_matrix(y_test, y_pred)

    # Normalize to percentages for easier reading
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0F0F1A')

    # ── Left: Raw counts ─────────────────────────────────────────────────────
    ax1 = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax1,
        linewidths=0.5,
        linecolor='#333355',
        cbar_kws={'shrink': 0.8}
    )
    ax1.set_title('Confusion Matrix\n(Raw Counts)',
                  color='white', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Predicted Label', color='#AAAAAA', fontsize=11, labelpad=10)
    ax1.set_ylabel('True Label',      color='#AAAAAA', fontsize=11, labelpad=10)
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#1A1A2E')

    # ── Right: Normalized percentages ───────────────────────────────────────
    ax2 = axes[1]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax2,
        linewidths=0.5,
        linecolor='#333355',
        cbar_kws={'shrink': 0.8, 'format': ticker.FuncFormatter(lambda x, _: f'{x:.0f}%')}
    )
    ax2.set_title('Confusion Matrix\n(Normalized %)',
                  color='white', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Predicted Label', color='#AAAAAA', fontsize=11, labelpad=10)
    ax2.set_ylabel('True Label',      color='#AAAAAA', fontsize=11, labelpad=10)
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#1A1A2E')

    fig.suptitle(
        'MNIST CNN — Confusion Matrix',
        color='white', fontsize=16, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"📊 Confusion matrix saved → {save_path}")


def plot_misclassified_examples(x_test, y_test, y_pred, n=20):
    """
    Plot examples that the model got wrong — useful for error analysis.

    Args:
        n (int): Number of misclassified examples to show.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Find indices of wrong predictions
    wrong_mask = (y_pred != y_test)
    wrong_idx  = np.where(wrong_mask)[0]

    if len(wrong_idx) == 0:
        print("🎉 No misclassifications found on the test set!")
        return

    # Sample up to n misclassified examples
    sample_idx = wrong_idx[:n]
    n_show     = len(sample_idx)
    cols, rows = 5, (n_show + 4) // 5

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    fig.patch.set_facecolor('#0F0F1A')
    fig.suptitle(
        f'Misclassified Examples (showing {n_show} of {wrong_mask.sum():,})',
        color='white', fontsize=14, fontweight='bold', y=1.01
    )

    for i, ax in enumerate(axes.flat):
        if i < n_show:
            idx   = sample_idx[i]
            ax.imshow(x_test[idx].squeeze(), cmap='gray')
            ax.set_title(
                f"True: {y_test[idx]}\nPred: {y_pred[idx]}",
                color='#FF6B6B', fontsize=9, fontweight='bold'
            )
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "misclassified.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    total_wrong = wrong_mask.sum()
    print(f"⚠️  Misclassified examples saved → {save_path}")
    print(f"   Total errors: {total_wrong:,} / {len(y_test):,}  "
          f"({total_wrong / len(y_test) * 100:.2f}% error rate)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained MNIST CNN model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   🧠 MNIST CNN — Model Evaluation")
    print("=" * 60 + "\n")

    # ── Load data & model ────────────────────────────────────────────────────
    x_test, y_test    = load_test_data()
    model             = load_model(args.model)

    # ── Run evaluation ───────────────────────────────────────────────────────
    y_pred, y_proba   = evaluate_model(model, x_test, y_test)
    acc               = print_evaluation_report(y_test, y_pred)

    # ── Generate plots ───────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred)
    plot_misclassified_examples(x_test, y_test, y_pred)

    print("\n✅ Evaluation complete!")
    print(f"   Final Test Accuracy: {acc*100:.2f}%")
    print(f"   Plots saved to: ./{PLOTS_DIR}/\n")


if __name__ == "__main__":
    main()
