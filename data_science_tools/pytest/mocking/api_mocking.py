from unittest.mock import Mock, patch

import requests


def fetch_stock_data(symbol):
	"""Fetch stock price data from an API"""
	response = requests.get(f"https://api.example.com/stock/{symbol}")
	return response.json()["price"]


@patch("requests.get")
def test_fetch_stock_data_simple(mock_get):
	# Create a fake response object
	mock_response = Mock()
	mock_response.json.return_value = {"price": 150.0}

	# Make the mock return our fake response
	mock_get.return_value = mock_response

	# Use the mock instead of real requests.get
	price = fetch_stock_data("AAPL")

	# Verify we got the fake data
	assert price == 150.0

	# Verify the mock was called with the right URL
	mock_get.assert_called_once_with("https://api.example.com/stock/AAPL")
