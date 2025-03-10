"""
hyperparameter_tuning.py

This module implements hyperparameter tuning for the CNN model using Keras Tuner.
The tuning process searches for the best hyperparameter configuration based on validation accuracy.
"""

import keras_tuner as kt
from src.model import create_cnn_model

def tune_model(x_train, y_train, x_val, y_val, max_trials=5, executions_per_trial=1, epochs=10):
    """
    Performs hyperparameter tuning using a Random Search strategy with Keras Tuner.
    
    Args:
        x_train: Training images.
        y_train: Training labels.
        x_val: Validation images.
        y_val: Validation labels.
        max_trials: Maximum number of hyperparameter configurations to try.
        executions_per_trial: Number of models to build and fit for each trial (to account for randomness).
        epochs: Number of epochs to train in each trial.
    
    returns:
        best_hps: The best hyperparameters found.
        tuner: The Keras Tuner object (contains detailed tuning results).
    """
    # Instantiate the RandomSearch tuner.
    tuner = kt.RandomSearch(
        hypermodel=create_cnn_model,       # The model-building function
        objective='val_accuracy',          # Optimize based on validation accuracy
        max_trials=max_trials,             # Try up to max_trials different hyperparameter combinations
        executions_per_trial=executions_per_trial,
        directory='kt_dir',                # Directory where the tuner will store its results
        project_name='cifar10_tuning'      # Project name (subdirectory in kt_dir)
    )
    
    # Start the hyperparameter search
    tuner.search(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
    
    # Retrieve the best hyperparameters from the tuning process.
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print the best hyperparameters to the console.
    print("Best hyperparameters found:", best_hps.values)
    return best_hps, tuner
