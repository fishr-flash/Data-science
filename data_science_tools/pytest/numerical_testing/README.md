# Numerical Testing Examples

This directory demonstrates proper testing techniques for numerical computations with NumPy arrays and pandas DataFrames.

## Files

- `numpy_arrays.py` - NumPy array testing with appropriate tolerance
- `dataframe_testing.py` - Pandas DataFrame testing utilities and patterns

## Key Concepts

### Floating-Point Precision Issues
Regular equality assertions often fail with floating-point numbers due to precision limitations. These tools solve that problem:

- `numpy.testing.assert_array_almost_equal()` - Compare arrays with tolerance
- `pandas.testing.assert_frame_equal()` - Compare DataFrames with configurable tolerance
- `pytest.approx()` - Compare individual floating-point values

### NumPy Testing Utilities
- `assert_array_almost_equal()` - Most common for approximate comparisons
- `assert_array_equal()` - For exact comparisons (integers, specific floats)
- `assert_allclose()` - Advanced tolerance control (relative and absolute)

### Pandas Testing Utilities
- `pd.testing.assert_frame_equal()` - Complete DataFrame comparison
- `pd.testing.assert_series_equal()` - Series comparison
- `pd.testing.assert_index_equal()` - Index comparison

## Running the Examples

```bash
# Run all numerical testing examples
pytest numerical_testing/

# Run specific test files
pytest numerical_testing/numpy_arrays.py
pytest numerical_testing/dataframe_testing.py

# Run with verbose output to see test descriptions
pytest -v numerical_testing/

# Run specific test functions
pytest numerical_testing/numpy_arrays.py::test_normalization
pytest numerical_testing/dataframe_testing.py::test_dataframe_cleaning
```

## Common Patterns

### Array Comparison with Tolerance
```python
# Bad - will often fail due to floating-point precision
assert result == expected

# Good - allows for small differences
np.testing.assert_array_almost_equal(result, expected, decimal=5)
```

### DataFrame Comparison
```python
# Compare entire DataFrames
pd.testing.assert_frame_equal(actual_df, expected_df)

# Compare with tolerance for floating-point columns
pd.testing.assert_frame_equal(actual_df, expected_df, atol=1e-5)

# Compare while ignoring index order
pd.testing.assert_frame_equal(
    actual_df.reset_index(drop=True), 
    expected_df.reset_index(drop=True)
)
```

### Statistical Property Testing
```python
# Test statistical properties instead of exact values
assert abs(data.mean()) < 0.1  # Should be close to 0
assert abs(data.std() - 1.0) < 0.1  # Should be close to 1
```

## Best Practices

1. **Choose appropriate tolerance**: Not too strict (causes flaky tests) or too loose (misses real errors)
2. **Test properties, not just values**: Mean, std, shape, data types
3. **Use parametrized tests**: Test with different array sizes, data types
4. **Handle edge cases**: NaN, infinity, empty arrays/DataFrames
5. **Test data types**: Ensure operations preserve or appropriately convert types
6. **Use reproducible random data**: Set seeds for consistent test results

## Data Science Applications

### Model Testing
- Compare model predictions with expected outputs
- Test feature engineering transformations
- Validate statistical properties of generated data

### Data Processing
- Test data cleaning operations
- Validate aggregation results
- Compare processed vs expected datasets

### Numerical Algorithms
- Test optimization algorithms convergence
- Validate mathematical transformations
- Compare algorithm implementations