PYTHON_DIRS = tests examples scripts eval_protocol

# Prefer tools from local virtualenv if present
VENV ?= .venv
VENV_BIN := $(VENV)/bin
PYTHON := $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,python)
FLAKE8 := $(if $(wildcard $(VENV_BIN)/flake8),$(VENV_BIN)/flake8,flake8)
MYPY := $(if $(wildcard $(VENV_BIN)/mypy),$(VENV_BIN)/mypy,mypy)
BLACK := $(if $(wildcard $(VENV_BIN)/black),$(VENV_BIN)/black,black)
PRE_COMMIT := $(if $(wildcard $(VENV_BIN)/pre-commit),$(VENV_BIN)/pre-commit,pre-commit)
PYTEST := $(if $(wildcard $(VENV_BIN)/pytest),$(VENV_BIN)/pytest,pytest)
TWINE := $(if $(wildcard $(VENV_BIN)/twine),$(VENV_BIN)/twine,twine)

.PHONY: clean build dist upload test lint typecheck format release sync-docs version tag-version show-version bump-major bump-minor bump-patch full-release quick-release pre-commit help

clean:
	rm -rf build/ dist/ *.egg-info/

# Run all pre-commit hooks (if installed)
pre-commit:
	$(PRE_COMMIT) run --all-files

build: clean
	$(PYTHON) -m build

dist: build

upload:
	$(TWINE) upload dist/*

test:
	$(PYTEST)

lint:
	$(PRE_COMMIT) run flake8 --all-files

typecheck:
	$(PRE_COMMIT) run mypy --all-files

format:
	$(PRE_COMMIT) run black --all-files && \
	$(PRE_COMMIT) run isort --all-files

validate-docs:
	@echo "Validating documentation links..."
	@if [ -f ~/home/docs/scripts/validate_links.py ]; then \
		cd ~/home/docs && python scripts/validate_links.py; \
	else \
		echo "❌ Error: Link validation script not found at ~/home/docs/scripts/validate_links.py"; \
		echo "Please ensure the validation script exists."; \
		exit 1; \
	fi

# Version management commands using versioneer
version:
	@echo "Current version information:"
	@python -c "import versioneer; print('Version:', versioneer.get_version())"
	@python -c "import versioneer; v = versioneer.get_versions(); print('Full info:', v)"

show-version:
	@python -c "import versioneer; print(versioneer.get_version())"

# Tag the current commit for release (creates git tag)
tag-version:
	@echo "Current version: $$(python -c 'import versioneer; print(versioneer.get_version())')"
	@read -p "Enter version to tag (e.g., 1.2.3): " version && \
		git tag -a "v$$version" -m "Release version $$version" && \
		echo "Tagged version v$$version"

# Helper commands for semantic versioning bumps
bump-patch:
	@current=$$(python -c "import versioneer; v=versioneer.get_version(); print(v.split('+')[0] if '+' in v else v)"); \
	if echo "$$current" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$$' > /dev/null; then \
		major=$$(echo $$current | cut -d. -f1); \
		minor=$$(echo $$current | cut -d. -f2); \
		patch=$$(echo $$current | cut -d. -f3); \
		next_patch=$$(( $$patch + 1 )); \
		next_version="$$major.$$minor.$$next_patch"; \
		echo "Current version: $$current"; \
		echo "Next patch version: $$next_version"; \
		read -p "Create tag v$$next_version? [y/N]: " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			git tag -a "v$$next_version" -m "Release version $$next_version" && \
			echo "Tagged version v$$next_version"; \
		fi; \
	else \
		echo "Current version ($$current) is not in semantic version format. Use 'make tag-version' instead."; \
	fi

bump-minor:
	@current=$$(python -c "import versioneer; v=versioneer.get_version(); print(v.split('+')[0] if '+' in v else v)"); \
	if echo "$$current" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$$' > /dev/null; then \
		major=$$(echo $$current | cut -d. -f1); \
		minor=$$(echo $$current | cut -d. -f2); \
		next_minor=$$(( $$minor + 1 )); \
		next_version="$$major.$$next_minor.0"; \
		echo "Current version: $$current"; \
		echo "Next minor version: $$next_version"; \
		read -p "Create tag v$$next_version? [y/N]: " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			git tag -a "v$$next_version" -m "Release version $$next_version" && \
			echo "Tagged version v$$next_version"; \
		fi; \
	else \
		echo "Current version ($$current) is not in semantic version format. Use 'make tag-version' instead."; \
	fi

bump-major:
	@current=$$(python -c "import versioneer; v=versioneer.get_version(); print(v.split('+')[0] if '+' in v else v)"); \
	if echo "$$current" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$$' > /dev/null; then \
		major=$$(echo $$current | cut -d. -f1); \
		next_major=$$(( $$major + 1 )); \
		next_version="$$next_major.0.0"; \
		echo "Current version: $$current"; \
		echo "Next major version: $$next_version"; \
		read -p "Create tag v$$next_version? [y/N]: " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			git tag -a "v$$next_version" -m "Release version $$next_version" && \
			echo "Tagged version v$$next_version"; \
		fi; \
	else \
		echo "Current version ($$current) is not in semantic version format. Use 'make tag-version' instead."; \
	fi

# Full release workflow with version tagging
full-release: lint typecheck test
	@echo "Current version: $$(python -c 'import versioneer; print(versioneer.get_version())')"
	@read -p "Enter new version to release (e.g., 1.2.3): " version && \
		git tag -a "v$$version" -m "Release version $$version" && \
		echo "Tagged version v$$version" && \
		$(MAKE) build && \
		$(MAKE) upload && \
		echo "Released version $$version to PyPI" && \
		echo "Don't forget to push the tag: git push origin v$$version"

# Quick release workflow (skips lint and typecheck)
quick-release: test
	@echo "⚠️  WARNING: Skipping lint and typecheck for quick release"
	@echo "Current version: $$(python -c 'import versioneer; print(versioneer.get_version())')"
	@read -p "Enter new version to release (e.g., 1.2.3): " version && \
		git tag -a "v$$version" -m "Release version $$version" && \
		echo "Tagged version v$$version" && \
		$(MAKE) build && \
		$(MAKE) upload && \
		echo "Released version $$version to PyPI" && \
		echo "Don't forget to push the tag: git push origin v$$version"

# This help target prints all available targets
help:
	@echo "Available targets:"
	@echo "  clean         - Remove build artifacts"
	@echo "  build         - Build source and wheel distributions"
	@echo "  dist          - Alias for build"
	@echo "  upload        - Upload to PyPI (make sure to bump version first)"
	@echo "  test          - Run tests"
	@echo "  lint          - Run flake8 via pre-commit"
	@echo "  typecheck     - Run mypy via pre-commit"
	@echo "  format        - Run black + isort via pre-commit"
	@echo "  validate-docs - Validate all documentation links in docs.json"
	@echo "  sync-docs     - Sync docs to ~/home/docs with links under 'evaluators'"
	@echo "  release       - Run lint, typecheck, test, build, then upload"
	@echo ""
	@echo "Version management (using versioneer):"
	@echo "  version       - Show current version information"
	@echo "  show-version  - Show current version string only"
	@echo "  tag-version   - Interactively create a git tag for release"
	@echo "  bump-patch    - Instructions for patch version bump"
	@echo "  bump-minor    - Instructions for minor version bump"
	@echo "  bump-major    - Instructions for major version bump"
	@echo "  full-release  - Full release workflow: test, tag, build, upload"
	@echo "  quick-release - Quick release workflow: test, tag, build, upload (skips lint/typecheck)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make version       - Check current version"
	@echo "  make tag-version   - Tag a new version"
	@echo "  make full-release  - Complete release process"
	@echo "  make quick-release - Fast release (skips lint/typecheck)"
	@echo "  make release       - Build and upload (assumes version already tagged)"
	@echo "  make lint          - Only run linting"
	@echo "  make format        - Format the code"
	@echo "  make sync-docs     - Sync documentation with path adjustments"

release: lint typecheck test build upload
	@echo "Published to PyPI"

# Demo for Remote Evaluation using Serveo.net
demo-remote-eval:
	@echo "---------------------------------------------------------------------"
	@echo "Running Remote Evaluation Demo with Serveo.net..."
	@echo "This demo will:"
	@echo "1. Generate a temporary API key."
	@echo "2. Start a local mock API service."
	@echo "3. Expose the mock API service to the internet using Serveo.net via SSH."
	@echo "   (Requires a working SSH client in your PATH)"
	@echo "4. Run evaluation functions that call the tunneled mock API service."
	@echo "5. Clean up all started processes on completion or interruption."
	@echo "---------------------------------------------------------------------"
	@echo "Log files for the demo will be created in ./logs/remote_eval_demo/"
	@echo "Starting demo script..."
	python examples/remote_eval_demo/run_demo.py
	@echo "---------------------------------------------------------------------"
	@echo "Remote Evaluation Demo finished."
	@echo "---------------------------------------------------------------------"
