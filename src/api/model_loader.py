import os
import tensorflow as tf
from fastapi import HTTPException

# Path to the trained CIFAR10 model
model_path = os.path.join(os.path.dirname(__file__), '../../cifar10_cnn_model.h5')

def load_model():
    """Loads the CIFAR10 CNN model and handles potential errors."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")