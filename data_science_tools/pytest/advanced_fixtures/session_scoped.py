import numpy as np
import pandas as pd
import pytest


# This fixture runs once per test session
@pytest.fixture(scope="session")
def large_dataset():
	# Simulate loading an expensive dataset
	print("Loading large dataset...")
	return pd.DataFrame({
		"feature1": np.random.randn(10000),
		"feature2": np.random.randn(10000),
		"target": np.random.randint(0, 2, 10000)
	})


def test_data_shape(large_dataset):
	assert large_dataset.shape == (10000, 3)


def test_feature_types(large_dataset):
	assert large_dataset["target"].dtype == int
	assert large_dataset["feature1"].dtype == float
