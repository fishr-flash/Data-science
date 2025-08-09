import numpy as np
import pandas as pd


def clean_dataframe(df):
	"""Remove duplicates and fill missing values"""
	return df.drop_duplicates().fillna(0)


def test_dataframe_cleaning():
	# Create test data with duplicates and NaN
	dirty_data = pd.DataFrame({"A": [1, 2, 2, np.nan], "B": [4, 5, 5, 6]})

	cleaned = clean_dataframe(dirty_data)

	expected = pd.DataFrame({"A": [1.0, 2.0, 0.0], "B": [4, 5, 6]})

	# Use pandas testing utility
	pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), expected)
