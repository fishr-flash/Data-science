import numpy as np
from numpy.testing import assert_array_almost_equal


def normalize_features(data):
	"""Normalize features to 0-1 range"""
	return (data - data.min()) / (data.max() - data.min())


def test_normalization():
	data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
	normalized = normalize_features(data)

	expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

	# Better than: assert normalized == expected (this would fail!)
	assert_array_almost_equal(normalized, expected, decimal=2)


def test_model_predictions():
	# Simulate model predictions with floating point results
	predictions = np.array([0.123456, 0.789012, 0.345678])
	expected = np.array([0.12, 0.79, 0.35])

	# Compare with 2 decimal places
	assert_array_almost_equal(predictions, expected, decimal=2)
