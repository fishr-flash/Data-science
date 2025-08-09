"""
Shared fixtures for advanced_fixtures examples.

This conftest.py file provides fixtures that can be used across
all test files in this directory without explicit imports.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def ml_dataset():
	"""Create a machine learning dataset for testing."""
	np.random.seed(42)  # For reproducibility

	# Generate features
	n_samples = 1000
	n_features = 4

	X = np.random.randn(n_samples, n_features)
	# Create a target variable with some relationship to features
	y = (X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, n_samples) > 0).astype(int)

	# Create DataFrame
	feature_names = [f"feature_{i + 1}" for i in range(n_features)]
	df = pd.DataFrame(X, columns=feature_names)
	df["target"] = y

	return df


@pytest.fixture(scope="module")
def data_processing_config():
	"""Configuration for data processing tests."""
	return {
		"train_size": 0.8,
		"random_state": 42,
		"normalize": True,
		"remove_outliers": True,
		"outlier_threshold": 3.0,
	}


@pytest.fixture
def sample_predictions():
	"""Generate sample model predictions for testing."""
	np.random.seed(42)
	return {
		"y_true": np.random.randint(0, 2, 100),
		"y_pred": np.random.rand(100),  # Probability predictions
		"y_pred_binary": np.random.randint(0, 2, 100),
	}


@pytest.fixture(autouse=True)
def reset_random_state():
	"""Ensure each test starts with a known random state."""
	np.random.seed(42)
	import random

	random.seed(42)
