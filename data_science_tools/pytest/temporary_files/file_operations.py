import pandas as pd


def save_model_predictions(predictions, filepath):
	"""Save model predictions to a CSV file"""
	import pandas as pd

	pd.DataFrame({"predictions": predictions}).to_csv(filepath, index=False)


def load_model_predictions(filepath):
	"""Load model predictions from a CSV file"""
	import pandas as pd

	return pd.read_csv(filepath)["predictions"].tolist()


def test_save_and_load_predictions(tmp_path):
	# tmp_path is automatically created and cleaned up
	predictions = [0.1, 0.9, 0.3, 0.7]

	# Create a temporary file path
	file_path = tmp_path / "predictions.csv"

	# Test saving
	save_model_predictions(predictions, file_path)
	assert file_path.exists()

	# Test loading
	loaded_predictions = load_model_predictions(file_path)
	assert loaded_predictions == predictions


def test_data_processing_pipeline(tmp_path):
	# Create temporary input file
	input_file = tmp_path / "input.csv"
	input_data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
	input_data.to_csv(input_file, index=False)

	# Create temporary output file path
	output_file = tmp_path / "processed.csv"

	# Test your processing function
	process_data(input_file, output_file)

	# Verify the output
	result = pd.read_csv(output_file)
	assert len(result) == 5
	# Add more specific assertions about your processing


def process_data(input_file, output_file):
	"""Simple data processing function for demonstration"""
	data = pd.read_csv(input_file)
	# Simple processing: multiply values by 2
	data["value"] = data["value"] * 2
	data.to_csv(output_file, index=False)
