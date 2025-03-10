"""
utils.py

Utility module that contains:
- Plot functions for training history (plot_history)
- Extended evaluation (evaluate_model) with confusion matrix and classification report
- Feature map visualization (visualize_feature_maps)
- Grad-CAM implementation (grad_cam, _grad_cam_model, display_grad_cam)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model

# ---------------------------------------
# 1. Plot Training History
# ---------------------------------------
def plot_history(history):
    """
    Plots the training and validation accuracy and loss from a model's history.
    The plots are saved to the 'results/' directory.

    Args:
        history: A History object returned by model.fit().
    """
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "accuracy.png"))
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "loss.png"))
    plt.close()

    print(f"Plots saved in directory: {results_dir}")

# ---------------------------------------
# 2. Extended Evaluation
# ---------------------------------------
def evaluate_model(model, x_test, y_test):
    """
    Prints a confusion matrix and classification report for the given model and test data.
    Also displays a confusion matrix as a heatmap.

    Args:
        model: Trained Keras model.
        x_test: Test images.
        y_test: Test labels.
    """
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = y_test.flatten()  # ensure shape is (n,)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    class_report = classification_report(y_true, y_pred)
    print(class_report)

# ---------------------------------------
# 3. Feature Map Visualization
# ---------------------------------------
def visualize_feature_maps(model, x_input, layer_index=0):
    """
    Visualizes the feature maps of a given convolutional layer for a single input image.

    Args:
        model: Trained Keras model.
        x_input: A single input image of shape (32, 32, 3) or a batch with shape (1, 32, 32, 3).
        layer_index: Index of the convolutional layer whose outputs we want to visualize.
    """
    # If the input is a single image (32, 32, 3), expand dims to (1, 32, 32, 3)
    if x_input.ndim == 3:
        x_input = np.expand_dims(x_input, axis=0)

    # Create a sub-model that outputs the activations of the chosen layer
    layer = model.layers[layer_index]
    activation_model = Model(inputs=model.input, outputs=layer.output)

    # Get the feature maps
    feature_maps = activation_model.predict(x_input)

    # Remove batch dimension if present (shape becomes (height, width, channels))
    feature_maps = np.squeeze(feature_maps, axis=0)

    # Number of channels in the feature map
    num_channels = feature_maps.shape[-1]

    # Determine grid size for plotting
    size = int(np.ceil(np.sqrt(num_channels)))

    fig, axes = plt.subplots(size, size, figsize=(12, 12))
    fig.suptitle(f"Feature Maps - Layer {layer_index} ({layer.name})", fontsize=16)

    channel_idx = 0
    for i in range(size):
        for j in range(size):
            ax = axes[i, j]
            if channel_idx < num_channels:
                ax.imshow(feature_maps[..., channel_idx], cmap='viridis')
                ax.axis('off')
                channel_idx += 1
            else:
                ax.remove()
    plt.show()

# ---------------------------------------
# 4. Grad-CAM
# ---------------------------------------
def grad_cam(model, image, layer_name, class_index=None):
    """
    Computes Grad-CAM for a specific class index (or the top predicted class if None).
    
    Args:
        model: Trained Keras model.
        image: A single image of shape (32, 32, 3) or (1, 32, 32, 3).
        layer_name: The name of the last convolutional layer to use for Grad-CAM.
        class_index: The class index for which Grad-CAM is computed. If None, uses the top prediction.
    
    Returns:
        heatmap: The Grad-CAM heatmap for the specified class.
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    if class_index is None:
        class_index = np.argmax(preds[0])

    class_output = model.output[:, class_index]
    last_conv_layer = model.get_layer(layer_name)

    with tf.GradientTape() as tape:
        # Create a sub-model that outputs (last_conv_layer_output, model.output)
        conv_outputs, predictions = _grad_cam_model(model, last_conv_layer.name)(tf.cast(image, tf.float32))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]

    # Global average pooling on the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    # Multiply each channel in the feature map by the corresponding gradient
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[..., i] *= pooled_grads[i]

    # Average the channels to get the heatmap
    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)
    return heatmap

def _grad_cam_model(model, layer_name):
    """
    Builds a sub-model that outputs (last_conv_layer_output, model.output).
    Used internally by grad_cam().
    """
    last_conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )
    return grad_model

def display_grad_cam(heatmap, original_image, alpha=0.4):
    """
    Displays the Grad-CAM heatmap overlayed on the original image.

    Args:
        heatmap: Grad-CAM heatmap (2D array).
        original_image: The original image array (32x32x3).
        alpha: Transparency factor for overlaying heatmap.
    """
    import cv2

    if original_image.ndim == 4:
        original_image = np.squeeze(original_image, axis=0)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (32, 32))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image.astype('uint8'), 1 - alpha, heatmap_colored, alpha, 0)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(original_image.astype('uint8'))
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(overlay)
    ax[1].set_title("Grad-CAM Overlay")
    ax[1].axis('off')

    plt.show()