"""
model.py

This module defines an advanced Convolutional Neural Network (CNN) architecture for CIFAR-10 classification.
It supports optional hyperparameter tuning via a Keras Tuner hyperparameter object (hp). The architecture
features multiple convolutional blocks with Batch Normalization, Dropout, and increased depth to create
a robust, high-performance model. The provided hyperparameter search space is designed to optimize key parameters
such that the default configuration is near-optimal and further fine-tuning is minimized.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(hp=None):
    """
    Creates and compiles an advanced CNN model for CIFAR-10 classification.

    The architecture consists of 4 convolutional blocks, each with:
      - Two Conv2D layers with ReLU activation and 'same' padding,
      - Batch Normalization after each convolution for stable and fast training,
      - A MaxPooling2D layer to reduce spatial dimensions,
      - Dropout for regularization.

    After the convolutional blocks, the feature maps are flattened and passed to a Dense layer
    (with Batch Normalization and Dropout), followed by the final output layer.

    Hyperparameter tuning (via the hp object from Keras Tuner) is supported for:
      - filters1: Number of filters in the first block,
      - multiplier: A factor to increase filters in the second block,
      - filters3 and filters4: Number of filters in block 3 and 4,
      - kernel_size: Kernel size for all convolutional layers (choice between 3 and 5),
      - dropout_conv: Dropout rate for each convolutional block,
      - dropout_dense: Dropout rate for the dense layer,
      - dense_units: Number of units in the dense layer,
      - learning_rate: Learning rate for the Adam optimizer.

    Args:
        hp: (optional) A hyperparameter object from Keras Tuner. If provided, the model's parameters
            will be tuned according to the specified ranges. Otherwise, default values are used.

    Returns:
        model: A compiled tf.keras.Model instance ready for training.
    """
    model = models.Sequential()

    # =======================
    # Set hyperparameters:
    # =======================
    if hp:
        # Number of filters in first block (e.g., 32 to 64 in steps of 16)
        filters1 = hp.Int('filters1', min_value=32, max_value=64, step=16, default=32)
        # Multiplier to increase filter count for block 2 (e.g., 1, 2, or 3)
        multiplier = hp.Choice('multiplier', values=[1, 2, 3], default=2)
        filters2 = filters1 * multiplier
        # For block 3, filters between filters2 and filters2*2
        filters3 = hp.Int('filters3', min_value=filters2, max_value=filters2 * 2, step=16, default=filters2)
        # For block 4, filters between filters3 and filters3*2
        filters4 = hp.Int('filters4', min_value=filters3, max_value=filters3 * 2, step=16, default=filters3)
        # Kernel size for convolutional layers: 3 or 5
        kernel_size = hp.Choice('kernel_size', values=[3, 5], default=3)
        # Dropout rate in each convolutional block
        dropout_conv = hp.Float('dropout_conv', min_value=0.1, max_value=0.5, step=0.1, default=0.25)
        # Dropout rate in the dense layer block
        dropout_dense = hp.Float('dropout_dense', min_value=0.3, max_value=0.7, step=0.1, default=0.5)
        # Number of units in the dense layer
        dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=64, default=256)
        # Learning rate for the optimizer, sampled logarithmically
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    else:
        filters1 = 32
        multiplier = 2
        filters2 = filters1 * multiplier  # 64
        filters3 = filters2               # default 64, can be tuned higher
        filters4 = filters3               # default 64, can be tuned higher
        kernel_size = 3
        dropout_conv = 0.25
        dropout_dense = 0.5
        dense_units = 256
        learning_rate = 1e-3

    # ============================================
    # Convolutional Block 1
    # ============================================
    # Input: 32x32 RGB images
    model.add(layers.Conv2D(filters1, (kernel_size, kernel_size), activation='relu', padding='same',
                            input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters1, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_conv))

    # ============================================
    # Convolutional Block 2
    # ============================================
    model.add(layers.Conv2D(filters2, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters2, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_conv))

    # ============================================
    # Convolutional Block 3
    # ============================================
    model.add(layers.Conv2D(filters3, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters3, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_conv))

    # ============================================
    # Convolutional Block 4
    # ============================================
    model.add(layers.Conv2D(filters4, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters4, (kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_conv))

    # ============================================
    # Fully Connected Layers
    # ============================================
    model.add(layers.Flatten())
    # Dense layer for high-level feature learning
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_dense))
    # Final output layer for classification into 10 classes
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model using Adam optimizer with the tuned learning rate,
    # sparse categorical crossentropy (suitable for integer class labels), and accuracy metric.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model