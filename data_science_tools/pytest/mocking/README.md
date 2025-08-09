# Mocking External Dependencies Examples

This directory demonstrates how to test data science code that depends on external services like APIs and databases without actually calling those services.

## Files

- `api_mocking.py` - Mock HTTP API calls (stock prices, weather data)
- `database_mocking.py` - Mock database queries and operations
- `requirements.txt` - Dependencies needed for mocking examples

## Key Concepts

### Why Mock External Dependencies?

1. **Speed**: Tests run faster without network calls or database connections
2. **Reliability**: Tests don't fail due to external service outages
3. **Cost**: Avoid charges from API calls during testing
4. **Control**: Test specific scenarios (errors, edge cases) that are hard to reproduce with real services
5. **Isolation**: Test your code logic independently of external services

### Mocking Tools

- `unittest.mock.patch` - Replace functions/methods with mock objects
- `unittest.mock.Mock` - Create mock objects with configurable behavior
- `side_effect` - Configure different responses for multiple calls
- `assert_called_with()` - Verify how mocks were called

## Running the Examples

```bash
# Install dependencies
pip install -r mocking/requirements.txt

# Run all mocking examples
pytest mocking/

# Run specific test files
pytest mocking/api_mocking.py
pytest mocking/database_mocking.py

# Run with verbose output to see mock interactions
pytest -v -s mocking/

# Run specific test functions
pytest mocking/api_mocking.py::test_portfolio_calculation
pytest mocking/database_mocking.py::test_sales_analysis
```

## Common Patterns

### Basic API Mocking
```python
@patch('requests.get')
def test_api_call(mock_get):
    # Configure mock response
    mock_response = Mock()
    mock_response.json.return_value = {'data': 'value'}
    mock_get.return_value = mock_response
    
    # Test your function
    result = your_function()
    
    # Verify mock was called correctly
    mock_get.assert_called_once_with('expected_url')
    assert result == expected_result
```

### Database Query Mocking
```python
@patch('pandas.read_sql')
def test_database_query(mock_read_sql):
    # Mock query result
    mock_data = pd.DataFrame({'col': [1, 2, 3]})
    mock_read_sql.return_value = mock_data
    
    # Test your function
    result = your_database_function()
    
    # Verify the query was correct
    mock_read_sql.assert_called_once_with(expected_query, connection)
```

### Multiple Responses
```python
@patch('requests.get')
def test_multiple_api_calls(mock_get):
    # Configure different responses for different calls
    responses = [
        Mock(json=lambda: {'price': 100}),
        Mock(json=lambda: {'price': 200})
    ]
    mock_get.side_effect = responses
    
    # Your function that makes multiple calls
    results = get_multiple_prices(['AAPL', 'GOOGL'])
```

## Data Science Applications

### API Testing
- **Financial data**: Stock prices, economic indicators
- **Weather data**: Climate data for agricultural models
- **Social media**: Tweet sentiment, social metrics
- **ML APIs**: Model inference, feature extraction

### Database Testing
- **Data warehouses**: Analytics queries, aggregations
- **Feature stores**: ML feature retrieval
- **Experiment tracking**: Model metrics, parameters
- **User data**: Customer analytics, behavior tracking

### Error Scenarios
- **Network failures**: Timeout, connection errors
- **API rate limits**: 429 responses, quota exceeded
- **Database locks**: Connection timeouts, deadlocks
- **Invalid responses**: Malformed data, missing fields

## Best Practices

1. **Mock at the right level**: Mock the external call, not internal logic
2. **Test error scenarios**: API failures, database errors, timeouts
3. **Verify interactions**: Check that external services are called correctly
4. **Use realistic data**: Mock responses should match real API/database schemas
5. **Test retry logic**: Ensure your code handles transient failures properly
6. **Avoid over-mocking**: Don't mock every function, focus on external dependencies

## Advanced Patterns

### Context Managers
```python
@patch('database.connection') 
def test_with_connection(mock_conn):
    # Mock database connection context manager
    mock_conn.return_value.__enter__.return_value = mock_connection
```

### Class-based Mocking
```python
@patch('your_module.ExternalAPIClient')
def test_api_client(mock_client_class):
    # Mock the entire class
    mock_instance = mock_client_class.return_value
    mock_instance.get_data.return_value = expected_data
```

### Partial Mocking
```python
# Mock only specific methods of a class
with patch.object(APIClient, 'authenticate', return_value=True):
    # authenticate() is mocked, other methods work normally
    client = APIClient()
    result = client.get_data()  # Real implementation
```