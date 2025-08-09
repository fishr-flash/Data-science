# conftest.py
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def sample_dataset():
	"""Create a sample dataset for testing"""
	np.random.seed(42)
	X = np.random.randn(1000, 5)
	y = np.random.randint(0, 2, 1000)

	return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)]).assign(target=y)


@pytest.fixture(scope="session")
def trained_model(sample_dataset):
	"""Provide a pre-trained model for testing"""
	X = sample_dataset.drop("target", axis=1)
	y = sample_dataset["target"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = LogisticRegression(random_state=42)
	model.fit(X_train, y_train)

	return {
		"model": model,
		"X_train": X_train,
		"X_test": X_test,
		"y_train": y_train,
		"y_test": y_test,
	}
