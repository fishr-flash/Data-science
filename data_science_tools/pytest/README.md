[![View on YouTube](https://img.shields.io/badge/YouTube-Watch%20on%20Youtube-red?logo=youtube)](https://www.youtube.com/playlist?list=PLnK6m_JBRVNoYEer9hBmTNwkYB3gmbOPO) [![View on Medium](https://img.shields.io/badge/Medium-View%20on%20Medium-blue?logo=medium)](https://towardsdatascience.com/pytest-for-data-scientists-2990319e55e6)

# Pytest for Data Scientists

Comprehensive examples and best practices for testing data science code with pytest.

## Directory Structure

```
pytest/
├── README.md                    # This file
├── get_started/                 # Basic pytest concepts
│   └── sentiment.py
├── parametrization/            # Parametrized testing
│   ├── process.py
│   ├── process_fixture.py
│   └── sentiment.py
├── test_structure_example/     # Project organization
│   ├── src/
│   └── tests/
├── advanced_fixtures/          # Advanced fixture patterns
│   ├── session_scoped.py
│   ├── autouse_fixtures.py
│   ├── conftest.py
│   └── README.md
├── temporary_files/           # Safe file I/O testing
│   ├── file_operations.py
│   ├── data_pipeline.py
│   └── README.md
├── numerical_testing/         # NumPy/DataFrame testing
│   ├── numpy_arrays.py
│   ├── dataframe_testing.py
│   └── README.md
├── mocking/                   # External dependency mocking
│   ├── api_mocking.py
│   ├── database_mocking.py
│   ├── requirements.txt
│   └── README.md
├── custom_markers/            # Test organization with markers
│   ├── pytest.ini
│   ├── marked_tests.py
│   └── README.md
└── project_config/           # Complete project configuration
    ├── pytest.ini
    ├── conftest.py
    ├── test_with_fixtures.py
    └── README.md
```

## Quick Start

### Basic Installation
```bash
pip install pytest

# For advanced features
pip install pytest-cov pytest-xdist pytest-benchmark
```

### Run Examples
```bash
# Basic examples
pytest get_started/
pytest parametrization/

# Advanced features
pytest advanced_fixtures/
pytest numerical_testing/
pytest mocking/

# Full project configuration
cd project_config && pytest
```

## Feature Overview

### 🚀 **Basic Concepts** (`get_started/`, `parametrization/`)
- Simple test functions and assertions
- Parametrized tests for multiple test cases
- Basic fixtures for data reuse

### 🔧 **Advanced Fixtures** (`advanced_fixtures/`)
- **Session-scoped fixtures**: Load expensive datasets once
- **Autouse fixtures**: Automatic setup for all tests  
- **Shared fixtures**: Common test data via `conftest.py`

### 📁 **Safe File Testing** (`temporary_files/`)
- **tmp_path fixture**: Isolated temporary directories
- **File I/O testing**: CSV, JSON, model serialization
- **Pipeline testing**: End-to-end data processing

### 🔢 **Numerical Testing** (`numerical_testing/`)
- **NumPy arrays**: Floating-point comparison with tolerance
- **Pandas DataFrames**: Proper DataFrame equality testing
- **Statistical validation**: Testing model outputs and data properties

### 🌐 **Mocking External Services** (`mocking/`)
- **API mocking**: Test without hitting real APIs
- **Database mocking**: Test queries without databases
- **Error simulation**: Test failure scenarios safely

### 🏷️ **Custom Markers** (`custom_markers/`)
- **Test organization**: Group tests by speed, requirements, domain
- **Selective execution**: Run specific test categories
- **CI/CD integration**: Different test suites for different stages

### ⚙️ **Project Configuration** (`project_config/`)
- **Complete setup**: Production-ready pytest configuration
- **Centralized fixtures**: Project-wide test utilities
- **Best practices**: Logging, warnings, reproducibility

## Common Workflows

### Development Workflow
```bash
# Fast feedback during development
pytest -m fast

# Before committing changes
pytest -m "fast or (integration and not slow)"

# Full test suite
pytest
```

### Continuous Integration
```bash
# Unit tests (fast feedback)
pytest -m "unit and fast"

# Integration tests  
pytest -m "integration and not gpu and not expensive"

# Performance tests (separate stage)
pytest -m "slow or expensive"
```

### Data Science Specific
```bash
# Test data processing pipelines
pytest -m data_processing

# Test model training
pytest -m model_training

# Test without external dependencies
pytest -m "not api and not database"
```

## Key Benefits for Data Scientists

### 🛡️ **Reliability**
- **Reproducible results**: Consistent random seeds
- **Isolated tests**: No interference between tests
- **Proper numerical comparison**: Handle floating-point precision

### ⚡ **Performance** 
- **Fast feedback**: Separate fast/slow test categories
- **Efficient fixtures**: Load expensive data once
- **Parallel execution**: Run tests concurrently

### 🔍 **Better Debugging**
- **Clear error messages**: Detailed assertion information
- **Test organization**: Easy to find and run specific tests
- **Comprehensive logging**: Track test execution

### 🤝 **Team Collaboration**
- **Standardized setup**: Consistent test environment
- **Shared fixtures**: Common test data and utilities  
- **Documentation**: Clear examples and best practices

## Testing Patterns by Use Case

### Data Processing
```python
def test_data_cleaning(tmp_path):
    # Use temporary files for safe testing
    input_file = tmp_path / "dirty_data.csv"
    # Test cleaning pipeline...
```

### Machine Learning
```python
@pytest.fixture(scope="session")
def trained_model():
    # Train once, test many aspects
    return expensive_model_training()

def test_model_accuracy(trained_model):
    # Test with proper numerical comparison
    assert model.accuracy > 0.9
```

### External APIs
```python
@patch('requests.get')
def test_api_integration(mock_get):
    # Mock external calls for reliable testing
    mock_get.return_value.json.return_value = {'data': 'test'}
    # Test your logic...
```

## Getting Help

Each directory contains detailed README files with:
- Specific feature documentation
- Running instructions  
- Best practices
- Troubleshooting guides

Start with the examples that match your current testing needs, then explore advanced features as your test suite grows.

## Related Resources

- **Article**: [Pytest for Data Scientists](https://towardsdatascience.com/pytest-for-data-scientists-2990319e55e6)
- **Video Series**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLnK6m_JBRVNoYEer9hBmTNwkYB3gmbOPO)
- **Official Docs**: [pytest.org](https://docs.pytest.org/)

## Contributing

These examples are designed to be practical and educational. Feel free to adapt them for your specific data science testing needs.