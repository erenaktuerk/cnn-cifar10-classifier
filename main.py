"""
main.py

This is the main entry point for the CIFAR-10 CNN Classification Project.
Running this script will initiate the full pipeline:
  - Data handling (saving and loading CIFAR-10 images)
  - (Optional) Hyperparameter tuning
  - Training of the CNN model
  - Saving the trained model and plots of training history
"""

from src.train import train_model

def main():
    # Print a starting message to the console.
    print("Starting the CIFAR-10 CNN training process...")
    
    # Call the training function which runs the entire training pipeline.
    train_model()
    
    # Print a completion message.
    print("Training process finished.")

if __name__ == "__main__":
    main()