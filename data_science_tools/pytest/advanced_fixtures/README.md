# Advanced Fixtures Examples

This directory demonstrates advanced pytest fixture patterns particularly useful for data science projects.

## Files

- `session_scoped.py` - Session-scoped fixtures for expensive operations (like loading large datasets)
- `autouse_fixtures.py` - Auto-use fixtures that run automatically before each test
- `conftest.py` - Shared fixtures available to all tests in this directory

## Key Concepts

### Session-Scoped Fixtures
- Run only once per test session
- Perfect for loading expensive datasets or training models
- Shared across all tests that request them
- Significant performance improvements for test suites

### Autouse Fixtures
- Automatically applied to all tests without explicit request
- Great for setup that should always happen (like setting random seeds)
- Ensures consistent test environments
- Reduces boilerplate code in individual tests

### Conftest.py
- Provides fixtures to all test files in the directory
- No imports needed - fixtures are automatically available
- Can have different conftest.py files at different directory levels
- Fixtures in parent directories are available to child directories

## Running the Examples

```bash
# Run all advanced fixture examples
pytest advanced_fixtures/

# Run with verbose output to see fixture setup
pytest -v advanced_fixtures/

# Run only session-scoped fixture tests
pytest advanced_fixtures/session_scoped.py

# Run only autouse fixture tests  
pytest advanced_fixtures/autouse_fixtures.py
```

## Key Benefits for Data Science

1. **Performance**: Session-scoped fixtures prevent reloading expensive datasets
2. **Reproducibility**: Autouse fixtures ensure consistent random seeds
3. **Organization**: Conftest.py centralizes common test data and setup
4. **Maintainability**: Reduces code duplication across test files