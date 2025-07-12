# Reward Protocol Tests

This directory contains tests specifically for verifying the `reward_protocol` dual import functionality.

## Test Files

### `test_reward_protocol_import.py`
- Comprehensive pytest-compatible test suite
- Tests import equivalence between `reward_kit` and `reward_protocol`
- Verifies that both packages provide identical functionality
- Tests require all dependencies to be installed

### `test_reward_protocol_simple.py`
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
- Tests that `reward_protocol` properly re-exports `reward_kit`
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
python tests/reward_protocol_tests/test_minimal_structure.py

# Run import equivalence tests (works in dev environment)
python tests/reward_protocol_tests/test_import_equivalence.py

# Run demo (works in dev environment)
python tests/reward_protocol_tests/demo_dual_imports.py

# Run simple tests (requires full installation)
python tests/reward_protocol_tests/test_reward_protocol_simple.py

# Run pytest tests (requires full installation)
pytest tests/test_reward_protocol_import.py -v
```

## Purpose

These tests verify that users can import the package using either:
- `from reward_kit import ...` (original style)
- `from reward_protocol import ...` (new style)

Both import styles provide identical functionality with no breaking changes. 