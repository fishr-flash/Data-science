from unittest.mock import patch

import pandas as pd

connection = None  # Simulated database connection


def get_sales_data(start_date, end_date):
	"""Fetch sales data from database"""
	query = f"SELECT * FROM sales WHERE date BETWEEN '{start_date}' AND '{end_date}'"
	return pd.read_sql(query, connection)


def analyze_sales_trends(start_date, end_date):
	"""Analyze sales trends over a period"""
	data = get_sales_data(start_date, end_date)
	return data.groupby("product")["amount"].sum().to_dict()


@patch("pandas.read_sql")
def test_sales_analysis(mock_read_sql):
	# Mock the database query result
	mock_data = pd.DataFrame(
		{
			"product": ["A", "B", "A", "B"],
			"amount": [100, 150, 200, 250],
			"date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
		}
	)
	mock_read_sql.return_value = mock_data

	result = analyze_sales_trends("2023-01-01", "2023-01-04")

	expected = {"A": 300, "B": 400}
	assert result == expected
