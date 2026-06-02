"""
train.py — Model Training Script
=================================
Author : Anas Dhafer Alzahrani
GitHub : https://github.com/Rflx9
Email  : anaas123190@gmail.com
---------------------------------
Loads the MNIST dataset, preprocesses it, trains the CNN model,
plots accuracy/loss curves, and saves the trained model to disk.

Usage:
    python train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import build_cnn_model

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EPOCHS      = 15          # Maximum number of training epochs
BATCH_SIZE  = 64          # Number of samples per gradient update
MODEL_PATH  = "mnist_cnn_model.keras"   # Where to save the trained model
PLOTS_DIR   = "plots"                   # Directory to save plots


def load_and_preprocess_data():
    """
    Load the MNIST dataset and preprocess it for training.

    Steps:
      1. Load raw pixel data (0–255) and integer labels
      2. Normalize pixel values to [0, 1] for faster convergence
      3. Reshape images to (28, 28, 1) — CNN expects a channel dimension

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    print("📦 Loading MNIST dataset...")

    # Load dataset — Keras auto-downloads it on first run (~11 MB)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"   Training samples : {len(x_train):,}")
    print(f"   Testing  samples : {len(x_test):,}")
    print(f"   Image shape      : {x_train.shape[1:]} (H × W)")
    print(f"   Pixel range      : [{x_train.min()} – {x_train.max()}]")

    # ── Normalize pixel values from [0, 255] → [0.0, 1.0] ──────────────────
    # Neural networks train faster and more stably with small input values
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ── Add channel dimension: (28, 28) → (28, 28, 1) ───────────────────────
    # Keras Conv2D layers expect shape (height, width, channels)
    # MNIST is grayscale, so channels = 1
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test,  axis=-1)

    print(f"\n✅ Preprocessing complete!")
    print(f"   x_train shape: {x_train.shape}")
    print(f"   x_test  shape: {x_test.shape}")
    print(f"   Pixel range after normalization: [{x_train.min():.1f} – {x_train.max():.1f}]")

    return x_train, y_train, x_test, y_test


def create_callbacks():
    """
    Create Keras training callbacks for smarter training.

    Callbacks:
      - ModelCheckpoint : Save the best model (by val_accuracy) automatically
      - EarlyStopping   : Stop early if validation accuracy stops improving
      - ReduceLROnPlateau: Lower learning rate when improvement stalls
    """
    os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else ".", exist_ok=True)

    callbacks = [
        # Save the model weights whenever validation accuracy improves
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # Stop training if val_accuracy doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate by 50% if val_loss plateaus for 3 epochs
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    return callbacks


def plot_training_history(history):
    """
    Generate and save training accuracy and loss plots.

    Args:
        history: Keras History object returned by model.fit()
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Extract metrics
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    # ── Create a clean, professional figure ─────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#0F0F1A')

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Shared style
    plot_style = {
        'facecolor': '#1A1A2E',
    }
    line_style = {'linewidth': 2.5, 'marker': 'o', 'markersize': 4}

    # ── Subplot 1: Accuracy ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(plot_style['facecolor'])
    ax1.plot(epochs, acc,     color='#00D4FF', label='Train Accuracy', **line_style)
    ax1.plot(epochs, val_acc, color='#FF6B6B', label='Val Accuracy',   **line_style, linestyle='--')
    ax1.set_title('Model Accuracy', color='white', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel('Epoch', color='#AAAAAA', fontsize=11)
    ax1.set_ylabel('Accuracy', color='#AAAAAA', fontsize=11)
    ax1.legend(facecolor='#2A2A3E', labelcolor='white', fontsize=10)
    ax1.tick_params(colors='#AAAAAA')
    ax1.spines[:].set_color('#333355')
    ax1.set_ylim([0.95, 1.0])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax1.grid(alpha=0.15, color='white')

    # ── Subplot 2: Loss ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(plot_style['facecolor'])
    ax2.plot(epochs, loss,     color='#00D4FF', label='Train Loss', **line_style)
    ax2.plot(epochs, val_loss, color='#FF6B6B', label='Val Loss',   **line_style, linestyle='--')
    ax2.set_title('Model Loss', color='white', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel('Epoch', color='#AAAAAA', fontsize=11)
    ax2.set_ylabel('Loss', color='#AAAAAA', fontsize=11)
    ax2.legend(facecolor='#2A2A3E', labelcolor='white', fontsize=10)
    ax2.tick_params(colors='#AAAAAA')
    ax2.spines[:].set_color('#333355')
    ax2.grid(alpha=0.15, color='white')

    fig.suptitle(
        'MNIST CNN — Training History',
        color='white', fontsize=16, fontweight='bold', y=1.02
    )

    save_path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊 Training plot saved → {save_path}")


def plot_sample_predictions(model, x_test, y_test):
    """
    Show a grid of sample test images with predicted vs true labels.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Pick 20 random test samples
    indices    = np.random.choice(len(x_test), 20, replace=False)
    images     = x_test[indices]
    true_labels = y_test[indices]

    # Get predictions
    preds       = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.patch.set_facecolor('#0F0F1A')
    fig.suptitle('Sample Predictions', color='white', fontsize=16,
                 fontweight='bold', y=0.98)

    for ax, img, true, pred in zip(axes.flat, images, true_labels, pred_labels):
        ax.imshow(img.squeeze(), cmap='gray')
        color  = '#00FF88' if pred == true else '#FF4444'
        title  = f"Pred: {pred}" if pred == true else f"Pred: {pred}\nTrue: {true}"
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"🖼️  Sample predictions plot saved → {save_path}")


def train():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("   🧠 MNIST Handwritten Digit Recognition — Training")
    print("=" * 60 + "\n")

    # ── Step 1: Load & preprocess data ───────────────────────────────────────
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # ── Step 2: Build model ──────────────────────────────────────────────────
    print("\n🔨 Building CNN model...")
    model = build_cnn_model()
    model.summary()

    # ── Step 3: Train ────────────────────────────────────────────────────────
    print(f"\n🚀 Starting training (max {EPOCHS} epochs, batch size {BATCH_SIZE})...")
    print("   Early stopping enabled — training stops if no improvement.\n")

    history = model.fit(
        x_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_split= 0.1,          # Use 10% of training data for validation
        callbacks       = create_callbacks(),
        verbose         = 1
    )

    # ── Step 4: Final evaluation ─────────────────────────────────────────────
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{'='*40}")
    print(f"  ✅ Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  📉 Test Loss     : {test_loss:.4f}")
    print(f"{'='*40}\n")

    # ── Step 5: Save plots ───────────────────────────────────────────────────
    plot_training_history(history)
    plot_sample_predictions(model, x_test, y_test)

    # ── Step 6: Confirm model saved ──────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"💾 Model saved → {MODEL_PATH}  ({size_mb:.1f} MB)")
    else:
        # Fallback save
        model.save(MODEL_PATH)
        print(f"💾 Model saved → {MODEL_PATH}")

    print("\n✅ Training complete! You can now run:")
    print("   python evaluate.py   — to evaluate the model")
    print("   streamlit run app.py — to launch the web app\n")


if __name__ == "__main__":
    train()
