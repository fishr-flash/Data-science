# Custom Markers for Test Organization

This directory demonstrates how to organize data science tests using custom pytest markers.

## Files

- `pytest.ini` - Configuration file defining custom markers
- `marked_tests.py` - Example tests with various marker combinations
- `README.md` - Documentation and usage instructions

## Available Markers

### Speed Categories
- `fast` - Quick unit tests (< 1 second)
- `slow` - Long-running tests (> 1 second)
- `expensive` - Computationally expensive tests

### Test Types
- `unit` - Unit tests for individual functions
- `integration` - Integration tests for multiple components
- `model_training` - Tests that train ML models
- `data_processing` - Tests for data transformation/cleaning

### Resource Requirements
- `gpu` - Tests requiring GPU/CUDA acceleration
- `api` - Tests requiring external API access
- `database` - Tests requiring database connections

### Dataset Categories
- `dataset_small` - Tests with small datasets (< 1000 rows)
- `dataset_large` - Tests with large datasets (> 10,000 rows)

## Running Tests by Marker

### Basic Usage
```bash
# Run only fast tests
pytest -m fast

# Run everything except slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run integration tests
pytest -m integration
```

### Complex Selection
```bash
# Run model training tests that are not slow
pytest -m "model_training and not slow"

# Run either API or database tests
pytest -m "api or database"

# Run fast unit tests for data processing
pytest -m "fast and unit and data_processing"

# Run integration tests that don't require GPU
pytest -m "integration and not gpu"
```

### List Available Markers
```bash
# See all configured markers
pytest --markers

# List tests with specific markers (dry run)
pytest --collect-only -m slow
```

## Typical Workflows

### Development Workflow
```bash
# Quick feedback loop - run only fast tests
pytest -m fast

# Before commit - run fast and some integration tests
pytest -m "fast or (integration and not slow)"
```

### CI/CD Pipelines
```bash
# Unit test stage (fast feedback)
pytest -m "unit and fast"

# Integration test stage
pytest -m "integration and not gpu and not expensive"

# Performance test stage
pytest -m "expensive or slow"

# GPU test stage (if GPU runners available)
pytest -m gpu
```

### Local Development by Feature
```bash
# Working on data processing features
pytest -m data_processing

# Working on model training
pytest -m model_training

# Testing database integration
pytest -m database
```

## Configuration Best Practices

### 1. Consistent Naming
Use clear, descriptive marker names that reflect:
- Speed: `fast`, `slow`, `expensive`
- Scope: `unit`, `integration`, `e2e`
- Requirements: `gpu`, `api`, `database`
- Domain: `model_training`, `data_processing`

### 2. Marker Combinations
Design markers to work well together:
```python
@pytest.mark.slow
@pytest.mark.model_training
@pytest.mark.gpu
def test_gpu_deep_learning():
    # This test is slow, trains models, and requires GPU
    pass
```

### 3. Documentation
Always document markers in `pytest.ini`:
```ini
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require GPU acceleration
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
# Fast tests on every PR
- name: Run unit tests
  run: pytest -m "unit and fast"

# Comprehensive tests on main branch
- name: Run integration tests  
  run: pytest -m "integration and not gpu"

# GPU tests on dedicated runners
- name: Run GPU tests
  run: pytest -m gpu
  if: runner.has-gpu
```

### Parallel Execution
```bash
# Run different marker groups in parallel
pytest -m "fast" --dist=loadscope &
pytest -m "slow" --dist=loadscope &
wait
```

## Troubleshooting

### Unknown Markers Warning
If you see warnings about unknown markers, ensure:
1. Markers are defined in `pytest.ini`
2. Use `--strict-markers` flag to catch typos
3. Check marker spelling in test files

### Marker Selection Not Working
- Use `-v` flag to see which tests are selected
- Use `--collect-only` to see test discovery without running
- Check boolean logic in marker expressions

### Performance Issues
- Profile marker selection: `pytest --collect-only -m "complex expression"`
- Simplify complex marker expressions
- Consider splitting tests into different files/directories

## Advanced Patterns

### Conditional Markers
```python
import sys
import pytest

# Skip GPU tests if CUDA not available
gpu_available = pytest.mark.skipif(
    not cuda_available(),
    reason="GPU/CUDA not available"
)

@gpu_available
@pytest.mark.gpu
def test_gpu_function():
    pass
```

### Dynamic Markers
```python
def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test characteristics."""
    for item in items:
        if "slow" in item.name or "train" in item.name:
            item.add_marker(pytest.mark.slow)
```

### Environment-based Selection
```bash
# Different marker sets for different environments
export TEST_ENV=development
pytest -m "fast and not integration"

export TEST_ENV=staging  
pytest -m "not gpu and not expensive"
```