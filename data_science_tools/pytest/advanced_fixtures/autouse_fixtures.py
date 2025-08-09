import numpy as np
import pytest


@pytest.fixture(autouse=True)
def setup_random_seeds():
	print("Setting up random seeds...")
	np.random.seed(42)
	import random
	random.seed(42)


def test_model_prediction():
	# This test will have reproducible random results
	X = np.random.randn(100, 5)
	# Your model training and prediction code here
	assert len(X) == 100


def test_data_sampling():
	# This test also gets reproducible randomness
	sample = np.random.choice([1, 2, 3, 4, 5], size=10)
	assert len(sample) == 10
