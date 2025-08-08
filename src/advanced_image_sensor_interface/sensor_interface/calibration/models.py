"""
Machine Learning Models for Sensor Calibration

This module provides machine learning models for automated sensor calibration
and parameter optimization.

Classes:
    CalibrationModel: Base class for calibration models.
    LinearCalibrationModel: Linear regression-based calibration model.
    NeuralCalibrationModel: Neural network-based calibration model.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CalibrationModel(ABC):
    """
    Abstract base class for calibration models.
    """

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the calibration model.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dict[str, Any]: Model information dictionary.
        """
        pass


class LinearCalibrationModel(CalibrationModel):
    """
    Linear regression-based calibration model.
    """

    def __init__(self):
        """Initialize the linear calibration model."""
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info("Linear calibration model initialized")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear calibration model.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
        """
        try:
            # Scale the input features
            X_scaled = self.scaler.fit_transform(X)

            # Train the model
            self.model.fit(X_scaled, y)
            self.is_trained = True

            logger.info(f"Linear model trained with {X.shape[0]} samples")
        except Exception as e:
            logger.error(f"Error training linear model: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained linear model.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the linear model.

        Returns:
            Dict[str, Any]: Model information dictionary.
        """
        info = {"model_type": "Linear Regression", "is_trained": self.is_trained}

        if self.is_trained:
            info.update(
                {
                    "coefficients": self.model.coef_.tolist() if hasattr(self.model, "coef_") else None,
                    "intercept": float(self.model.intercept_) if hasattr(self.model, "intercept_") else None,
                    "score": getattr(self.model, "score_", None),
                }
            )

        return info


class NeuralCalibrationModel(CalibrationModel):
    """
    Neural network-based calibration model.
    """

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (100, 50), max_iter: int = 1000):
        """
        Initialize the neural calibration model.

        Args:
            hidden_layer_sizes (Tuple[int, ...]): Sizes of hidden layers.
            max_iter (int): Maximum number of iterations for training.
        """
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.hidden_layer_sizes = hidden_layer_sizes
        logger.info(f"Neural calibration model initialized with layers: {hidden_layer_sizes}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the neural calibration model.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
        """
        try:
            # Scale the input features
            X_scaled = self.scaler.fit_transform(X)

            # Train the model
            self.model.fit(X_scaled, y)
            self.is_trained = True

            logger.info(f"Neural model trained with {X.shape[0]} samples, " f"converged in {self.model.n_iter_} iterations")
        except Exception as e:
            logger.error(f"Error training neural model: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained neural model.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the neural model.

        Returns:
            Dict[str, Any]: Model information dictionary.
        """
        info = {
            "model_type": "Neural Network (MLP)",
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "is_trained": self.is_trained,
        }

        if self.is_trained:
            info.update(
                {
                    "n_layers": self.model.n_layers_,
                    "n_iter": self.model.n_iter_,
                    "loss": float(self.model.loss_) if hasattr(self.model, "loss_") else None,
                }
            )

        return info


# Example usage
if __name__ == "__main__":
    # Generate sample calibration data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    # Create a non-linear relationship for testing
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(n_samples)

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Test linear model
    linear_model = LinearCalibrationModel()
    linear_model.train(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)
    linear_mse = np.mean((y_test - linear_predictions) ** 2)

    print(f"Linear Model MSE: {linear_mse:.4f}")
    print(f"Linear Model Info: {linear_model.get_model_info()}")

    # Test neural model
    neural_model = NeuralCalibrationModel(hidden_layer_sizes=(50, 25))
    neural_model.train(X_train, y_train)
    neural_predictions = neural_model.predict(X_test)
    neural_mse = np.mean((y_test - neural_predictions) ** 2)

    print(f"Neural Model MSE: {neural_mse:.4f}")
    print(f"Neural Model Info: {neural_model.get_model_info()}")
