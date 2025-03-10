"""
train.py

This module orchestrates the full training process of the CIFAR-10 CNN model.
It performs the following steps:
  1. Saves the CIFAR-10 images to disk for future reference.
  2. Loads and preprocesses the data.
  3. Splits the training data to create a validation set.
  4. Optionally runs hyperparameter tuning.
  5. Trains the model using EarlyStopping to prevent overfitting.
  6. Saves the trained model.
  7. Plots and saves the training history (accuracy and loss curves).
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from src.data_preprocessing import load_and_preprocess_data, save_cifar10_to_disk
from src.model import create_cnn_model
from src.utils import plot_history, evaluate_model
from src.hyperparameter_tuning import tune_model

def train_model():
    """
    Trains the CNN model on the CIFAR-10 dataset with data augmentation,
    hyperparameter tuning, and extended evaluation.
    """
    # Step 1: Save CIFAR-10 images to disk (if needed)
    save_cifar10_to_disk()

    # Step 2: Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Create a validation split
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train_new, y_train_new = x_train[val_size:], y_train[val_size:]

    # Step 3: (Optional) Hyperparameter tuning
    DO_TUNING = True
    if DO_TUNING:
        print("Starting hyperparameter tuning...")
        best_hps, tuner = tune_model(x_train_new, y_train_new, x_val, y_val, max_trials=5, epochs=5)
        model = create_cnn_model(best_hps)
        print("Best hyperparameters found:", best_hps.values)
    else:
        model = create_cnn_model()

    # Step 4: Set up data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rotation_range=15,      # random rotations up to 15°
        width_shift_range=0.1,  # horizontal shifts
        height_shift_range=0.1, # vertical shifts
        horizontal_flip=True,   # random horizontal flips
        fill_mode='nearest'     # fill in missing pixels
    )
    # Important: fit the generator on the training data
    train_datagen.fit(x_train_new)

    # Step 5: Train the model using the augmented data generator
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        train_datagen.flow(x_train_new, y_train_new, batch_size=32),
        epochs=10,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # Step 6: Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # Step 7: Plot training history
    plot_history(history)

    # Step 8: Extended evaluation (Confusion Matrix & Classification Report)
    evaluate_model(model, x_test, y_test)

    # Step 9: save the trained model
    model.save('cifar10_cnn_model.h5')
    print("Model training complete and saved.")

if __name__ == "__main__":
    train_model()