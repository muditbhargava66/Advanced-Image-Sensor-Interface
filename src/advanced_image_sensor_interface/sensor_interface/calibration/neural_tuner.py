"""
Neural Tuner for Advanced Image Sensor Interface

This module implements AI-based parameter tuning for optimal sensor performance
using neural networks and optimization algorithms.

Classes:
    NeuralTuner: Main class for AI-based parameter tuning.
"""

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .models import CalibrationModel, NeuralCalibrationModel

logger = logging.getLogger(__name__)


class NeuralTuner:
    """
    AI-based parameter tuning system for image sensor optimization.

    This class uses neural networks and optimization algorithms to automatically
    tune sensor parameters for optimal performance.
    """

    def __init__(self, parameter_ranges: dict[str, tuple[float, float]]):
        """
        Initialize the NeuralTuner.

        Args:
            parameter_ranges (Dict[str, Tuple[float, float]]): Dictionary mapping
                parameter names to their (min, max) ranges.
        """
        self.parameter_ranges = parameter_ranges
        self.parameter_names = list(parameter_ranges.keys())
        self.model: CalibrationModel = NeuralCalibrationModel(hidden_layer_sizes=(64, 32, 16))
        self.training_data: list[tuple[dict[str, float], float]] = []
        self.is_trained = False
        logger.info(f"Neural Tuner initialized with parameters: {self.parameter_names}")

    def add_training_sample(self, parameters: dict[str, float], performance_score: float) -> None:
        """
        Add a training sample to the dataset.

        Args:
            parameters (Dict[str, float]): Parameter values.
            performance_score (float): Performance score for these parameters.
        """
        # Validate parameters
        for param_name, value in parameters.items():
            if param_name not in self.parameter_ranges:
                raise ValueError(f"Unknown parameter: {param_name}")

            min_val, max_val = self.parameter_ranges[param_name]
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter {param_name} value {value} out of range [{min_val}, {max_val}]")

        self.training_data.append((parameters.copy(), performance_score))
        logger.debug(f"Added training sample: {parameters} -> {performance_score}")

    def train_model(self) -> None:
        """Train the neural network model on collected data."""
        if len(self.training_data) < 10:
            raise ValueError("Need at least 10 training samples to train the model")

        # Convert training data to arrays
        X = []
        y = []

        for parameters, score in self.training_data:
            # Create feature vector in consistent order
            feature_vector = [parameters.get(param_name, 0.0) for param_name in self.parameter_names]
            X.append(feature_vector)
            y.append(score)

        X = np.array(X)
        y = np.array(y)

        # Normalize features to [0, 1] range
        X_normalized = np.zeros_like(X)
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameter_ranges[param_name]
            X_normalized[:, i] = (X[:, i] - min_val) / (max_val - min_val)

        # Train the model
        self.model.train(X_normalized, y)
        self.is_trained = True
        logger.info(f"Model trained on {len(self.training_data)} samples")

    def predict_performance(self, parameters: dict[str, float]) -> float:
        """
        Predict performance score for given parameters.

        Args:
            parameters (Dict[str, float]): Parameter values.

        Returns:
            float: Predicted performance score.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Create and normalize feature vector
        feature_vector = [parameters.get(param_name, 0.0) for param_name in self.parameter_names]
        X = np.array([feature_vector])

        X_normalized = np.zeros_like(X)
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameter_ranges[param_name]
            X_normalized[:, i] = (X[:, i] - min_val) / (max_val - min_val)

        prediction = self.model.predict(X_normalized)
        return float(prediction[0])

    def optimize_parameters(self, initial_guess: dict[str, float] | None = None) -> dict[str, float]:
        """
        Find optimal parameters using the trained model.

        Args:
            initial_guess (Dict[str, float], optional): Initial parameter guess.
                If None, uses midpoint of parameter ranges.

        Returns:
            Dict[str, float]: Optimized parameter values.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before optimization")

        # Set initial guess
        if initial_guess is None:
            initial_guess = {}
            for param_name, (min_val, max_val) in self.parameter_ranges.items():
                initial_guess[param_name] = (min_val + max_val) / 2

        # Convert to normalized array
        x0 = []
        bounds = []
        for param_name in self.parameter_names:
            min_val, max_val = self.parameter_ranges[param_name]
            value = initial_guess.get(param_name, (min_val + max_val) / 2)
            x0.append((value - min_val) / (max_val - min_val))
            bounds.append((0.0, 1.0))

        x0 = np.array(x0)

        # Define objective function (negative because we want to maximize performance)
        def objective(x_normalized):
            try:
                prediction = self.model.predict(x_normalized.reshape(1, -1))
                return -float(prediction[0])  # Negative for maximization
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 1e6  # Large positive value for minimization

        # Optimize
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        # Convert back to original parameter space
        optimal_params = {}
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameter_ranges[param_name]
            normalized_value = result.x[i]
            optimal_params[param_name] = min_val + normalized_value * (max_val - min_val)

        predicted_score = -result.fun
        logger.info(f"Optimization completed. Predicted score: {predicted_score:.4f}")
        logger.info(f"Optimal parameters: {optimal_params}")

        return optimal_params

    def get_tuner_status(self) -> dict[str, Any]:
        """
        Get the current status of the neural tuner.

        Returns:
            Dict[str, Any]: Status information dictionary.
        """
        return {
            "parameter_names": self.parameter_names,
            "parameter_ranges": self.parameter_ranges,
            "training_samples": len(self.training_data),
            "is_trained": self.is_trained,
            "model_info": self.model.get_model_info() if self.is_trained else None,
        }


# Example usage
if __name__ == "__main__":
    # Define parameter ranges for image sensor tuning
    parameter_ranges = {
        "gain": (1.0, 16.0),
        "exposure_time": (0.001, 1.0),
        "noise_reduction": (0.0, 1.0),
        "sharpening": (0.0, 2.0),
        "color_saturation": (0.5, 2.0),
    }

    tuner = NeuralTuner(parameter_ranges)

    # Generate synthetic training data
    np.random.seed(42)
    for _ in range(50):
        # Random parameters within ranges
        params = {}
        for param_name, (min_val, max_val) in parameter_ranges.items():
            params[param_name] = np.random.uniform(min_val, max_val)

        # Synthetic performance function (higher is better)
        # This would be replaced with actual sensor performance measurement
        score = (
            100
            - abs(params["gain"] - 4.0) * 5  # Optimal gain around 4.0
            + (1.0 - params["exposure_time"]) * 20  # Prefer shorter exposure
            + params["noise_reduction"] * 30  # More noise reduction is better
            + (2.0 - abs(params["sharpening"] - 1.0)) * 15  # Optimal sharpening around 1.0
            + (2.0 - abs(params["color_saturation"] - 1.2)) * 10  # Optimal saturation around 1.2
            + np.random.normal(0, 5)  # Add some noise
        )

        tuner.add_training_sample(params, score)

    # Train the model
    tuner.train_model()

    # Find optimal parameters
    optimal_params = tuner.optimize_parameters()
    predicted_score = tuner.predict_performance(optimal_params)

    print(f"Optimal parameters: {optimal_params}")
    print(f"Predicted performance score: {predicted_score:.2f}")

    # Print tuner status
    status = tuner.get_tuner_status()
    print(f"Tuner status: {status}")
