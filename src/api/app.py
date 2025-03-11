from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
from src.api.model_loader import load_model

# Debug message to confirm app.py is executed
print("🔥 DEBUG: app.py is being executed!")

# Create FastAPI instance
app = FastAPI()

# Debug message to show registered routes before adding any
print(f"🚧 Registered routes before adding: {app.routes}")

# Load the model and confirm model loading
try:
    model = load_model()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")

# Data model for the API input
class InputData(BaseModel):
    features: list  # Flattened list of 32x32x3 pixel values

    @validator("features")
    def validate_features(cls, value):
        """Validates that the input is a flattened list of 32x32x3 pixel values."""
        expected_length = 32 * 32 * 3
        if len(value) != expected_length:
            raise ValueError(f"Invalid input shape. Expected {expected_length} pixel values, got {len(value)}.")
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("All feature values must be numeric.")
        return value

# Root endpoint to confirm the API is running
@app.get("/")
async def root():
    return {"message": "🚀 CIFAR10 API is running!"}

# Debug message after adding the root route
print(f"✅ Root route added. Current routes: {app.routes}")

# Endpoint to list all registered routes
@app.get("/routes")
async def get_routes():
    return [{"path": route.path, "methods": route.methods} for route in app.routes]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Convert input data to NumPy array and reshape to (1, 32, 32, 3)
        input_data = np.array(data.features).reshape(1, 32, 32, 3)

        # Normalize pixel values to the [0, 1] range
        input_data = input_data.astype('float32') / 255.0

        # Get model predictions
        prediction = model.predict(input_data)

        # Get the class with the highest probability
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        # Return prediction results
        return {
            "predicted_class": predicted_class,
            "probabilities": prediction.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Final debug message showing all routes after adding them
print(f"🚦 Final registered routes: {app.routes}")