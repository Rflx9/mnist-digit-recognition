"""
model.py — CNN Architecture for Handwritten Digit Recognition
=============================================================
Author : Anas Dhafer Alzahrani
GitHub : https://github.com/Rflx9
Email  : anaas123190@gmail.com
-------------------------------------------------------------
Defines the Convolutional Neural Network (CNN) model used to classify
handwritten digits (0–9) from the MNIST dataset.

Architecture Overview:
  Input → Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build and return a compiled CNN model for digit recognition.

    Args:
        input_shape (tuple): Shape of each input image. Default is (28, 28, 1)
                             for grayscale MNIST images.
        num_classes (int):   Number of output classes. Default is 10 (digits 0–9).

    Returns:
        model: A compiled Keras Sequential model.
    """

    model = models.Sequential(name="MNIST_CNN")

    # ──────────────────────────────────────────────
    # Block 1 — First Convolutional Block
    # ──────────────────────────────────────────────
    # Conv2D: Learn 32 different 3×3 feature detectors (edges, curves, etc.)
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',          # ReLU removes negative values (adds non-linearity)
        input_shape=input_shape,
        padding='same',             # Keep spatial dimensions the same
        name='conv_1'
    ))

    # MaxPooling: Downsample by taking the max value in each 2×2 region
    # This reduces computation and makes the model more translation-invariant
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool_1'))

    # ──────────────────────────────────────────────
    # Block 2 — Second Convolutional Block (deeper features)
    # ──────────────────────────────────────────────
    # 64 filters: learns more complex patterns (loops, intersections, etc.)
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv_2'
    ))

    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool_2'))

    # ──────────────────────────────────────────────
    # Block 3 — Third Convolutional Block (high-level features)
    # ──────────────────────────────────────────────
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv_3'
    ))

    # ──────────────────────────────────────────────
    # Classifier Head
    # ──────────────────────────────────────────────
    # Flatten: Convert 3D feature maps → 1D vector for Dense layers
    model.add(layers.Flatten(name='flatten'))

    # Dropout: Randomly zero out 50% of neurons during training
    # This prevents overfitting (model memorizing training data)
    model.add(layers.Dropout(0.5, name='dropout'))

    # Dense: Fully connected layer — combines all features
    model.add(layers.Dense(128, activation='relu', name='dense_1'))

    # Output layer: 10 neurons (one per digit 0–9)
    # Softmax converts raw scores to probabilities that sum to 1
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))

    # ──────────────────────────────────────────────
    # Compile the Model
    # ──────────────────────────────────────────────
    model.compile(
        optimizer='adam',                       # Adaptive learning rate optimizer
        loss='sparse_categorical_crossentropy', # Loss for integer class labels
        metrics=['accuracy']                    # Track accuracy during training
    )

    return model


def get_model_summary(model):
    """Print a detailed summary of the model architecture."""
    print("\n" + "=" * 60)
    print("         CNN MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    model.summary()
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Quick test: build and display the model
    model = build_cnn_model()
    get_model_summary(model)
    print(f"✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
