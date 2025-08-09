# Temporary Files Testing Examples

This directory demonstrates how to safely test file operations using pytest's `tmp_path` fixture.

## Files

- `file_operations.py` - Basic file I/O operations (CSV, JSON, model serialization)
- `data_pipeline.py` - Data processing pipeline testing with temporary files

## Key Concepts

### tmp_path Fixture
- Automatically creates a temporary directory for each test
- Directory and all contents are cleaned up after the test
- Safe to create, modify, and delete files without affecting real data
- Each test gets its own isolated temporary directory

### Benefits for Data Science
1. **Safety**: No risk of corrupting real data files
2. **Isolation**: Tests don't interfere with each other
3. **Cleanup**: No manual cleanup needed
4. **Realistic**: Test with actual file operations, not mocks

## Running the Examples

```bash
# Run all temporary files examples
pytest temporary_files/

# Run with verbose output to see test names
pytest -v temporary_files/

# Run specific test file
pytest temporary_files/file_operations.py
pytest temporary_files/data_pipeline.py

# Run specific test function
pytest temporary_files/file_operations.py::test_save_and_load_predictions
```

## Example Use Cases

### File Operations Testing
- Model serialization/deserialization (pickle, joblib)
- Data format conversions (CSV, JSON, Parquet)
- Configuration file handling
- Log file generation

### Data Pipeline Testing
- ETL pipeline components
- Data cleaning and preprocessing
- Dataset splitting and sampling
- File-based data validation

### Error Handling
- Missing file scenarios
- Permission errors
- Corrupted file handling
- Invalid format detection

## Best Practices

1. **Use tmp_path for all file operations**: Never test with real files
2. **Test both success and failure cases**: Include error scenarios
3. **Verify file contents**: Don't just check file existence
4. **Test file formats**: Ensure data integrity across different formats
5. **Test edge cases**: Empty files, large files, special characters