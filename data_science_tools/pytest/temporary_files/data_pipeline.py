from pathlib import Path

import numpy as np
import pandas as pd


def process_data(input_file, output_file):
	"""Process data from input file and save to output file."""
	# Load data
	df = pd.read_csv(input_file)

	# Basic data processing
	processed_df = df.copy()

	# Add a processed column (example transformation)
	if "value" in df.columns:
		processed_df["value_squared"] = df["value"] ** 2
		processed_df["value_normalized"] = (df["value"] - df["value"].mean()) / df[
			"value"
		].std()

	# Remove any rows with missing values
	processed_df = processed_df.dropna()

	# Save processed data
	processed_df.to_csv(output_file, index=False)

	return len(processed_df)


def clean_dataset(input_path, output_path, remove_duplicates=True, fill_na=True):
	"""Clean a dataset and save the cleaned version."""
	df = pd.read_csv(input_path)

	# Remove duplicates if requested
	if remove_duplicates:
		df = df.drop_duplicates()

	# Fill missing values if requested
	if fill_na:
		# Fill numeric columns with median
		numeric_columns = df.select_dtypes(include=[np.number]).columns
		for col in numeric_columns:
			df[col] = df[col].fillna(df[col].median())

		# Fill string columns with mode
		string_columns = df.select_dtypes(include=["object"]).columns
		for col in string_columns:
			df[col] = df[col].fillna(
				df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
			)

	# Save cleaned data
	df.to_csv(output_path, index=False)

	return {
		"original_rows": len(pd.read_csv(input_path)),
		"cleaned_rows": len(df),
		"removed_rows": len(pd.read_csv(input_path)) - len(df),
	}


def split_dataset(input_path, output_dir, train_ratio=0.8, random_state=42):
	"""Split a dataset into train/test sets."""
	df = pd.read_csv(input_path)

	# Shuffle the data
	df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

	# Calculate split point
	split_point = int(len(df_shuffled) * train_ratio)

	# Split the data
	train_df = df_shuffled[:split_point]
	test_df = df_shuffled[split_point:]

	# Save splits
	output_dir = Path(output_dir)
	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"

	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	return {
		"train_path": train_path,
		"test_path": test_path,
		"train_size": len(train_df),
		"test_size": len(test_df),
	}


# Test functions


def test_data_processing_pipeline(tmp_path):
	"""Test the complete data processing pipeline."""
	# Create temporary input file
	input_file = tmp_path / "input.csv"
	input_data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
	input_data.to_csv(input_file, index=False)

	# Create temporary output file path
	output_file = tmp_path / "processed.csv"

	# Test the processing function
	rows_processed = process_data(input_file, output_file)

	# Verify the output file exists
	assert output_file.exists()

	# Verify the processing worked correctly
	result = pd.read_csv(output_file)
	assert len(result) == 5
	assert rows_processed == 5

	# Check that new columns were added
	assert "value_squared" in result.columns
	assert "value_normalized" in result.columns

	# Verify transformations
	expected_squared = [1, 4, 9, 16, 25]
	assert result["value_squared"].tolist() == expected_squared


def test_dataset_cleaning(tmp_path):
	"""Test dataset cleaning functionality."""
	# Create dirty dataset
	dirty_data = pd.DataFrame(
		{
			"name": ["Alice", "Bob", "Alice", "Charlie", None],
			"age": [25, 30, 25, 35, np.nan],
			"score": [85.5, 90.0, 85.5, 95.0, 88.0],
		}
	)

	input_path = tmp_path / "dirty_data.csv"
	output_path = tmp_path / "clean_data.csv"

	dirty_data.to_csv(input_path, index=False)

	# Clean the dataset
	cleaning_stats = clean_dataset(
		input_path, output_path, remove_duplicates=True, fill_na=True
	)

	# Verify output exists
	assert output_path.exists()

	# Load and verify cleaned data
	cleaned = pd.read_csv(output_path)

	# Should have removed one duplicate row (Alice)
	assert len(cleaned) == 4
	assert cleaning_stats["original_rows"] == 5
	assert cleaning_stats["cleaned_rows"] == 4
	assert cleaning_stats["removed_rows"] == 1

	# Should have no missing values
	assert cleaned.isnull().sum().sum() == 0

	# Verify missing age was filled with median
	assert cleaned["age"].tolist() == [
		25.0,
		30.0,
		35.0,
		30.0,
	]  # median of [25,30,35] is 30


def test_dataset_splitting(tmp_path):
	"""Test dataset splitting functionality."""
	# Create sample dataset
	sample_data = pd.DataFrame(
		{
			"feature1": np.random.randn(100),
			"feature2": np.random.randn(100),
			"target": np.random.randint(0, 2, 100),
		}
	)

	input_path = tmp_path / "full_dataset.csv"
	sample_data.to_csv(input_path, index=False)

	# Split the dataset
	split_info = split_dataset(input_path, tmp_path, train_ratio=0.8, random_state=42)

	# Verify split files exist
	assert split_info["train_path"].exists()
	assert split_info["test_path"].exists()

	# Verify split sizes
	assert split_info["train_size"] == 80
	assert split_info["test_size"] == 20
	assert split_info["train_size"] + split_info["test_size"] == 100

	# Load and verify split data
	train_df = pd.read_csv(split_info["train_path"])
	test_df = pd.read_csv(split_info["test_path"])

	assert len(train_df) == 80
	assert len(test_df) == 20

	# Verify all original columns are preserved
	original_columns = set(sample_data.columns)
	assert set(train_df.columns) == original_columns
	assert set(test_df.columns) == original_columns


def test_pipeline_with_missing_input(tmp_path):
	"""Test pipeline behavior with missing input files."""
	import pytest

	# Test with non-existent input file
	missing_input = tmp_path / "missing.csv"
	output_file = tmp_path / "output.csv"

	with pytest.raises(FileNotFoundError):
		process_data(missing_input, output_file)


def test_pipeline_with_empty_dataset(tmp_path):
	"""Test pipeline behavior with empty dataset."""
	# Create empty dataset
	empty_data = pd.DataFrame(columns=["value"])
	input_path = tmp_path / "empty.csv"
	output_path = tmp_path / "empty_processed.csv"

	empty_data.to_csv(input_path, index=False)

	# Process empty dataset
	rows_processed = process_data(input_path, output_path)

	# Should handle empty dataset gracefully
	assert rows_processed == 0
	assert output_path.exists()

	result = pd.read_csv(output_path)
	assert len(result) == 0
	assert "value_squared" in result.columns  # Column should still be created
