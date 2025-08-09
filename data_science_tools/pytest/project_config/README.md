# Project Configuration Examples

This directory demonstrates comprehensive pytest configuration for data science projects, including centralized fixtures and project-wide settings.

## Files

- `pytest.ini` - Complete pytest configuration with data science-specific settings
- `conftest.py` - Centralized fixtures and test configuration
- `test_with_fixtures.py` - Examples using the shared fixtures
- `README.md` - Documentation and best practices

## Key Configuration Features

### pytest.ini Settings
- **Test discovery**: Automatic test file/function detection
- **Custom markers**: Organized markers for data science workflows  
- **Output formatting**: Verbose reporting with timing information
- **Warning filters**: Suppression of common ML library warnings
- **Logging configuration**: Detailed test logging setup
- **Timeout settings**: Prevent hanging tests

### conftest.py Fixtures
- **Session-scoped**: Expensive operations like dataset creation
- **Module-scoped**: Model training that can be shared across test files
- **Function-scoped**: Individual test data and mocks
- **Autouse**: Automatic seed setting for reproducibility

## Running the Examples

### Basic Usage
```bash
# Run all tests using project configuration
pytest project_config/

# Run with project configuration from any directory
cd project_config && pytest

# Run specific test file
pytest project_config/test_with_fixtures.py
```

### Using Custom Markers
```bash
# Run only fast tests
pytest -m fast project_config/

# Run integration tests
pytest -m integration project_config/

# Run data processing tests
pytest -m data_processing project_config/
```

### Advanced Options
```bash
# Run with coverage (if pytest-cov installed)
pytest --cov=src project_config/

# Run with timing information
pytest --durations=10 project_config/

# Run with detailed logging
pytest --log-cli-level=DEBUG project_config/
```

## Configuration Benefits

### 1. Consistency Across Team
- Standardized test behavior across all developers
- Common fixtures available to all tests
- Consistent random seeds for reproducible results

### 2. Performance Optimization
- Session-scoped fixtures prevent repeated expensive operations
- Proper warning suppression reduces noise
- Timeout settings prevent infinite hanging tests

### 3. Better Debugging
- Detailed logging configuration for troubleshooting
- Timing information to identify slow tests
- Clear test failure reporting

### 4. CI/CD Integration
- Markers enable selective test execution in pipelines
- Consistent behavior across different environments
- Proper exit codes and reporting

## Fixture Design Patterns

### Session-Scoped Data
```python
@pytest.fixture(scope="session")
def large_dataset():
    """Load once, use many times."""
    return load_expensive_dataset()
```

### Model Training
```python
@pytest.fixture(scope="module")
def trained_model(dataset):
    """Train once per test module."""
    return train_model(dataset)
```

### Test Utilities
```python
@pytest.fixture
def evaluation_metrics():
    """Provide reusable evaluation functions."""
    return {'accuracy': accuracy_score, 'f1': f1_score}
```

### Autouse Setup
```python
@pytest.fixture(autouse=True)
def reproducible_tests():
    """Ensure every test has consistent random state."""
    np.random.seed(42)
```

## Best Practices

### 1. Fixture Organization
- **Session**: Expensive data loading, model training
- **Module**: Shared between tests in same file
- **Function**: Individual test setup
- **Autouse**: Universal setup (seeds, warnings)

### 2. Marker Strategy
- **Speed**: `fast`, `slow`, `expensive`
- **Scope**: `unit`, `integration`, `e2e`  
- **Requirements**: `gpu`, `api`, `database`
- **Domain**: `model_training`, `data_processing`

### 3. Configuration Management
- Keep `pytest.ini` in project root
- Use environment-specific overrides when needed
- Document all custom markers and their usage
- Regularly review and update warning filters

### 4. Fixture Dependencies
```python
@pytest.fixture
def processed_data(raw_data):
    """Build fixtures on top of other fixtures."""
    return preprocess(raw_data)

@pytest.fixture  
def model_results(trained_model, test_data):
    """Combine multiple fixtures."""
    return trained_model.predict(test_data)
```

## Integration with Development Workflow

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-fast
      name: Run fast tests
      entry: pytest -m fast
      language: system
```

### IDE Integration
Most IDEs automatically detect and use pytest.ini configuration:
- Test discovery follows configured paths
- Markers appear in test runners
- Fixtures are available in autocomplete

### CI/CD Pipeline
```yaml
# GitHub Actions example
- name: Run unit tests
  run: pytest -m "fast and unit"
  
- name: Run integration tests  
  run: pytest -m "integration and not gpu"
  
- name: Generate coverage
  run: pytest --cov=src --cov-report=xml
```

## Troubleshooting

### Common Issues

**Fixtures not found**: Check `conftest.py` location and import paths
**Markers not recognized**: Ensure markers are defined in `pytest.ini`  
**Tests running slowly**: Review fixture scopes and session-scoped data
**Random failures**: Check for proper seed setting in autouse fixtures

### Debugging Commands
```bash
# See all available fixtures
pytest --fixtures project_config/

# See all markers
pytest --markers

# Dry run to see test discovery
pytest --collect-only project_config/

# Debug specific fixture
pytest --fixtures-per-test project_config/test_with_fixtures.py::test_name
```

### Performance Optimization
```bash
# Profile test execution
pytest --durations=0 project_config/

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto project_config/

# Skip slow tests during development  
pytest -m "not slow" project_config/
```