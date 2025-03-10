from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

# Path to the trained CIFAR10 model
model_path = os.path.join(os.path.dirname(__file__), '../../cifar10_cnn_model.h5')

# Load the model with error handling
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

# Create FastAPI instance
app = FastAPI()
print(app.routes)

# Data model for the API input
class InputData(BaseModel):
    features: list  # Flattened list of 32x32x3 pixel values

# Root endpoint to confirm the API is running
@app.get("/")
async def root():
    return {"message": "cifar10-API is running!"}

@app.get("/hello")
async def hello():
    return {"message": "Hello, world!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Validate input shape
        expected_length = 32 * 32 * 3
        if len(data.features) != expected_length:
            raise HTTPException(status_code=400, detail=f"Invalid input shape. Expected {expected_length} pixel values.")

        # Convert input data to NumPy array and reshape to (1, 32, 32, 3)
        input_data = np.array(data.features).reshape(1, 32, 32, 3)

        # Normalize pixel values to the [0, 1] range
        input_data = input_data.astype('float32') / 255.0

        # Get model predictions
        prediction = model.predict(input_data)

        # Get the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Return prediction results
        return {
            "predicted_class": int(predicted_class),
            "probabilities": prediction.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")