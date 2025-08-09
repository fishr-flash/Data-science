"""
Example tests demonstrating the use of shared fixtures from conftest.py.

These tests show how to use the various fixtures defined in conftest.py
for testing data science workflows.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ========== TESTS USING DATASET FIXTURES ==========


def test_sample_dataset_properties(sample_dataset):
	"""Test the properties of the sample dataset fixture."""
	assert isinstance(sample_dataset, pd.DataFrame)
	assert len(sample_dataset) == 1000
	assert "target" in sample_dataset.columns

	# Check feature columns
	feature_cols = [col for col in sample_dataset.columns if col.startswith("feature_")]
	assert len(feature_cols) == 5

	# Check target distribution (should be roughly balanced)
	target_counts = sample_dataset["target"].value_counts()
	assert len(target_counts) == 2  # Binary classification
	assert min(target_counts) > 300  # Both classes well represented


def test_regression_dataset_properties(regression_dataset):
	"""Test the properties of the regression dataset fixture."""
	assert isinstance(regression_dataset, pd.DataFrame)
	assert len(regression_dataset) == 800
	assert "target" in regression_dataset.columns

	# Check for continuous target (regression)
	assert regression_dataset["target"].dtype in ["float64", "float32"]

	# Target should have reasonable variance
	assert regression_dataset["target"].std() > 0.5


def test_time_series_data_properties(time_series_data):
	"""Test the time series data fixture."""
	assert isinstance(time_series_data, pd.DataFrame)
	assert len(time_series_data) == 730  # 2 years of daily data

	# Check required columns
	required_cols = ["date", "value", "day_of_week", "month"]
	for col in required_cols:
		assert col in time_series_data.columns

	# Check date range
	assert time_series_data["date"].min() == pd.Timestamp("2022-01-01")
	assert time_series_data["date"].max() == pd.Timestamp("2023-12-31")

	# Check day_of_week and month ranges
	assert time_series_data["day_of_week"].min() >= 0
	assert time_series_data["day_of_week"].max() <= 6
	assert time_series_data["month"].min() >= 1
	assert time_series_data["month"].max() <= 12


# ========== TESTS USING MODEL FIXTURES ==========


def test_trained_classifier_structure(trained_classifier):
	"""Test the structure of the trained classifier fixture."""
	assert "models" in trained_classifier
	assert "X_train" in trained_classifier
	assert "X_test" in trained_classifier
	assert "y_train" in trained_classifier
	assert "y_test" in trained_classifier
	assert "feature_names" in trained_classifier

	# Check models
	models = trained_classifier["models"]
	assert "logistic" in models
	assert "random_forest" in models

	# Check data shapes
	X_train = trained_classifier["X_train"]
	y_train = trained_classifier["y_train"]
	assert len(X_train) == len(y_train)
	assert X_train.shape[1] == 5  # 5 features


def test_model_predictions(trained_classifier):
	"""Test that trained models can make predictions."""
	models = trained_classifier["models"]
	X_test = trained_classifier["X_test"]

	for _model_name, model in models.items():
		# Test predictions
		predictions = model.predict(X_test)
		probabilities = model.predict_proba(X_test)

		# Basic checks
		assert len(predictions) == len(X_test)
		assert len(probabilities) == len(X_test)
		assert probabilities.shape[1] == 2  # Binary classification

		# Predictions should be 0 or 1
		assert set(predictions).issubset({0, 1})

		# Probabilities should sum to 1
		np.testing.assert_array_almost_equal(probabilities.sum(axis=1), 1.0)


def test_model_evaluation(trained_classifier, model_evaluation_metrics):
	"""Test model evaluation using the metrics fixture."""
	models = trained_classifier["models"]
	X_test = trained_classifier["X_test"]
	y_test = trained_classifier["y_test"]
	evaluate_classifier = model_evaluation_metrics["evaluate_classifier"]

	for _model_name, model in models.items():
		predictions = model.predict(X_test)
		metrics = evaluate_classifier(y_test, predictions)

		# Check that all metrics are present
		required_metrics = ["accuracy", "precision", "recall", "f1"]
		for metric in required_metrics:
			assert metric in metrics
			assert 0 <= metrics[metric] <= 1  # Metrics should be between 0 and 1

		# Model should perform better than random
		assert metrics["accuracy"] > 0.6


# ========== TESTS USING UTILITY FIXTURES ==========


def test_small_dataframe_fixture(small_dataframe):
	"""Test the small DataFrame fixture."""
	assert isinstance(small_dataframe, pd.DataFrame)
	assert len(small_dataframe) == 5
	assert list(small_dataframe.columns) == ["A", "B", "C", "D"]

	# Test specific data types
	assert small_dataframe["A"].dtype in ["int64", "int32"]
	assert small_dataframe["D"].dtype in ["float64", "float32"]
	assert small_dataframe["C"].dtype == object  # string column


def test_mock_api_client_fixture(mock_api_client):
	"""Test the mock API client fixture."""
	# Test get_data method
	data_response = mock_api_client.get_data()
	assert data_response["status"] == "success"
	assert len(data_response["data"]) == 2

	# Test predictions method
	pred_response = mock_api_client.get_model_predictions()
	assert "predictions" in pred_response
	assert "model_version" in pred_response
	assert len(pred_response["predictions"]) == 4

	# Test health check
	health_response = mock_api_client.health_check()
	assert health_response["status"] == "healthy"


def test_temporary_directory_fixture(temporary_directory):
	"""Test the temporary directory fixture."""
	assert isinstance(temporary_directory, Path)
	assert temporary_directory.exists()
	assert temporary_directory.is_dir()

	# Create a file in the temp directory
	test_file = temporary_directory / "test.txt"
	test_file.write_text("Hello, World!")

	assert test_file.exists()
	assert test_file.read_text() == "Hello, World!"


def test_sample_model_artifacts_fixture(sample_model_artifacts):
	"""Test the sample model artifacts fixture."""
	artifacts = sample_model_artifacts

	# Check directory exists
	assert artifacts["directory"].exists()
	assert artifacts["directory"].is_dir()

	# Check all files exist
	assert artifacts["model_file"].exists()
	assert artifacts["config_file"].exists()
	assert artifacts["metadata_file"].exists()

	# Check file contents
	assert "mock_model_content" in artifacts["model_file"].read_text()
	assert "param1" in artifacts["config_file"].read_text()
	assert "version" in artifacts["metadata_file"].read_text()


# ========== TESTS DEMONSTRATING AUTOUSE FIXTURES ==========


def test_random_seed_reproducibility():
	"""Test that random seeds are reset for reproducibility."""
	# Generate random numbers - should be the same every time due to autouse fixture
	random_array1 = np.random.randn(5)

	# In a separate test, we should get different numbers, but within this test,
	# multiple calls should be deterministic
	expected_first_value = random_array1[0]

	# Reset seed manually to verify fixture behavior
	np.random.seed(42)
	random_array2 = np.random.randn(5)

	# First values should match due to seed reset
	assert random_array2[0] == expected_first_value


# ========== TESTS WITH MULTIPLE FIXTURES ==========


@pytest.mark.integration
def test_complete_ml_pipeline(
	sample_dataset, trained_classifier, model_evaluation_metrics
):
	"""Test a complete ML pipeline using multiple fixtures."""
	# Get original dataset
	original_data = sample_dataset

	# Get trained models
	models = trained_classifier["models"]
	X_test = trained_classifier["X_test"]
	y_test = trained_classifier["y_test"]

	# Get evaluation function
	evaluate_classifier = model_evaluation_metrics["evaluate_classifier"]

	# Test the complete pipeline
	results = {}

	for model_name, model in models.items():
		# Make predictions
		predictions = model.predict(X_test)

		# Evaluate model
		metrics = evaluate_classifier(y_test, predictions)
		results[model_name] = metrics

	# Compare models
	assert len(results) == 2

	# Both models should have reasonable performance
	for _model_name, metrics in results.items():
		assert metrics["accuracy"] > 0.6
		assert metrics["f1"] > 0.6

	# Random Forest might perform better than Logistic Regression
	# (but this isn't guaranteed with small dataset)
	rf_accuracy = results["random_forest"]["accuracy"]
	lr_accuracy = results["logistic"]["accuracy"]

	# Both should be reasonable
	assert rf_accuracy > 0.5
	assert lr_accuracy > 0.5


@pytest.mark.data_processing
def test_data_processing_pipeline(sample_dataset, small_dataframe):
	"""Test data processing using multiple fixture datasets."""

	def process_dataset(df):
		"""Simple data processing function."""
		processed = df.copy()

		# Add some engineered features for numeric columns
		numeric_cols = df.select_dtypes(include=[np.number]).columns
		for col in numeric_cols:
			if col != "target":  # Don't process target variable
				processed[f"{col}_squared"] = df[col] ** 2
				processed[f"{col}_log"] = np.log(np.abs(df[col]) + 1)

		return processed

	# Process both datasets
	processed_sample = process_dataset(sample_dataset)
	processed_small = process_dataset(small_dataframe)

	# Check that processing worked
	assert len(processed_sample.columns) > len(sample_dataset.columns)
	assert len(processed_small.columns) > len(small_dataframe.columns)

	# Check for engineered features
	assert any("_squared" in col for col in processed_sample.columns)
	assert any("_log" in col for col in processed_sample.columns)


# ========== PARAMETRIZED TESTS WITH FIXTURES ==========


@pytest.mark.parametrize("model_name", ["logistic", "random_forest"])
def test_individual_model_performance(
	trained_classifier, model_evaluation_metrics, model_name
):
	"""Test individual model performance using parametrization."""
	model = trained_classifier["models"][model_name]
	X_test = trained_classifier["X_test"]
	y_test = trained_classifier["y_test"]
	evaluate_classifier = model_evaluation_metrics["evaluate_classifier"]

	predictions = model.predict(X_test)
	metrics = evaluate_classifier(y_test, predictions)

	# Each model should meet minimum performance standards
	assert metrics["accuracy"] > 0.5
	assert metrics["f1"] > 0.4
	assert metrics["precision"] > 0.4
	assert metrics["recall"] > 0.4
