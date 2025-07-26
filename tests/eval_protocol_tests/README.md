# Reward Protocol Tests

This directory contains tests specifically for verifying the `eval_protocol` dual import functionality.

## Test Files

### `test_eval_protocol_import.py`
- Comprehensive pytest-compatible test suite
- Tests import equivalence between `eval_protocol` and `eval_protocol`
- Verifies that both packages provide identical functionality
- Tests require all dependencies to be installed

### `test_eval_protocol_simple.py`
- Basic import tests that work with a fully installed environment
- Tests that both packages can be imported and provide the same functionality
- Verifies version consistency and API equivalence

### `test_minimal_structure.py`
- Structure verification tests that work without dependencies
- Tests package discovery, file structure, and configuration
- Verifies that the build process includes both packages
- ✅ **Passes in development environment**

### `test_import_equivalence.py`
- Tests import equivalence by examining module structure
- Verifies console scripts and package metadata
- Tests that `eval_protocol` properly re-exports `eval_protocol`
- ✅ **Passes in development environment**

### `demo_dual_imports.py`
- Demonstration script showing how dual imports work
- Shows usage examples for both import styles
- Provides migration guide for users
- Verifies package structure and configuration

## Running the Tests

### From the project root:
```bash
# Run minimal structure tests (works in dev environment)
python tests/eval_protocol_tests/test_minimal_structure.py

# Run import equivalence tests (works in dev environment)
python tests/eval_protocol_tests/test_import_equivalence.py

# Run demo (works in dev environment)
python tests/eval_protocol_tests/demo_dual_imports.py

# Run simple tests (requires full installation)
python tests/eval_protocol_tests/test_eval_protocol_simple.py

# Run pytest tests (requires full installation)
pytest tests/test_eval_protocol_import.py -v
```

## Purpose

These tests verify that users can import the package using either:
- `from eval_protocol import ...` (original style)
- `from eval_protocol import ...` (new style)

Both import styles provide identical functionality with no breaking changes. 