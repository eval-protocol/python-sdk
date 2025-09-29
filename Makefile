PYTHON_DIRS = tests examples scripts eval_protocol
PY ?= uv run python

.PHONY: clean build dist upload test lint typecheck format release sync-docs version tag-version show-version bump-major bump-minor bump-patch full-release quick-release
## -----------------------------
## Local Langfuse + LiteLLM E2E
## -----------------------------

.PHONY: local-install local-langfuse-up local-langfuse-up-local local-langfuse-wait local-litellm-up local-litellm-smoke local-adapter-smoke local-generate-traces local-generate-chinook local-eval local-eval-fireworks-only local-quick-run

local-install:
	uv pip install -e ".[langfuse]"

# 1) Start Langfuse per official docs (run from Langfuse repo). Here we just export env.
local-langfuse-up:
	@echo "Ensure you started Langfuse via docker compose as per docs."
	@echo "Docs: https://langfuse.com/self-hosting/deployment/docker-compose"
	@echo "Exporting LANGFUSE env vars for SDK..."
	LANGFUSE_PUBLIC_KEY=$${LANGFUSE_PUBLIC_KEY:-local}; \
	LANGFUSE_SECRET_KEY=$${LANGFUSE_SECRET_KEY:-local}; \
	LANGFUSE_HOST=$${LANGFUSE_HOST:-http://localhost:3000}; \
	printf "LANGFUSE_PUBLIC_KEY=%s\nLANGFUSE_SECRET_KEY=%s\nLANGFUSE_HOST=%s\n" $$LANGFUSE_PUBLIC_KEY $$LANGFUSE_SECRET_KEY $$LANGFUSE_HOST

# Start Langfuse using local compose file
local-langfuse-up-local:
	docker compose -f examples/local_langfuse_litellm_ollama/langfuse-docker-compose.yml up -d

# Wait until Langfuse UI responds
local-langfuse-wait:
	LANGFUSE_HOST=$${LANGFUSE_HOST:-http://localhost:3000}; \
	echo "Waiting for $$LANGFUSE_HOST ..."; \
	for i in $$(seq 1 60); do \
	  code=$$(curl -s -o /dev/null -w "%{http_code}" $$LANGFUSE_HOST); \
	  if [ "$$code" = "200" ] || [ "$$code" = "302" ]; then echo "Langfuse is up (HTTP $$code)"; exit 0; fi; \
	  sleep 2; \
	done; \
	echo "Langfuse did not become ready in time."; exit 1

# 2) Start LiteLLM router (requires litellm installed). Keep foreground.
local-litellm-up:
	LITELLM_API_KEY=$${LITELLM_API_KEY:-local-demo-key}; \
	printf "LITELLM_API_KEY=%s\n" $$LITELLM_API_KEY; \
	LITELLM_API_KEY=$$LITELLM_API_KEY uv run litellm --config examples/local_langfuse_litellm_ollama/litellm-config.yaml --port 4000

# 2b) Smoke test LiteLLM endpoints
local-litellm-smoke:
	@test -n "$$LITELLM_API_KEY" || (echo "LITELLM_API_KEY not set" && exit 1)
	curl -s -H "Authorization: Bearer $$LITELLM_API_KEY" http://127.0.0.1:4000/v1/models | head -n 5 | cat
	curl -s \
	  -H "Authorization: Bearer $$LITELLM_API_KEY" \
	  -H "Content-Type: application/json" \
	  http://127.0.0.1:4000/v1/chat/completions \
	  -d '{"model":"ollama/llama3.1","messages":[{"role":"user","content":"Say hi"}]}' \
	| head -n 40 | cat

# 3) Seed one trace into Langfuse

# 4) Adapter smoke test (fetch 1 row)
local-adapter-smoke:
	LANGFUSE_HOST=$${LANGFUSE_HOST:-http://localhost:3000}; \
	code=$$(curl -s -o /dev/null -w "%{http_code}" $$LANGFUSE_HOST); \
	if [ "$$code" != "200" ] && [ "$$code" != "302" ]; then \
	  echo "Langfuse not reachable at $$LANGFUSE_HOST (HTTP $$code). Start it per docs."; \
	  exit 1; \
	fi; \
	LANGFUSE_PUBLIC_KEY=$${LANGFUSE_PUBLIC_KEY:-local}; \
	LANGFUSE_SECRET_KEY=$${LANGFUSE_SECRET_KEY:-local}; \
	LANGFUSE_PUBLIC_KEY=$$LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY=$$LANGFUSE_SECRET_KEY LANGFUSE_HOST=$$LANGFUSE_HOST \
	$(PY) -c "from eval_protocol.adapters.langfuse import create_langfuse_adapter; a=create_langfuse_adapter(); rows=a.get_evaluation_rows(limit=1, sample_size=1); print('Fetched rows:', len(rows))"

# Generate realistic traces into Langfuse (Chinook) using Fireworks models
local-generate-traces:
	@test -n "$$FIREWORKS_API_KEY" || (echo "FIREWORKS_API_KEY not set" && exit 1)
	uv pip install -e ".[pydantic,fireworks,chinook]" >/dev/null || true
	CHINOOK_USE_STUB_DB=1 uv run pytest tests/chinook/langfuse/generate_traces.py -q

# Force-run Chinook generator with stub DB and Langfuse observe
local-generate-chinook:
	@test -n "$$FIREWORKS_API_KEY" || (echo "FIREWORKS_API_KEY not set" && exit 1)
	uv pip install -e ".[pydantic,fireworks,chinook]" >/dev/null || true
	CHINOOK_USE_STUB_DB=1 uv run pytest tests/chinook/langfuse/generate_traces.py -q

# Fallback generator that does not need external DBs

# 5) Run the local evaluation test (uses Fireworks as judge; requires FIREWORKS_API_KEY)
local-eval:
	@test -n "$$FIREWORKS_API_KEY" || (echo "FIREWORKS_API_KEY not set" && exit 1)
	uv run pytest eval_protocol/quickstart/llm_judge_langfuse_local.py -k test_llm_judge_local -q

# Run evaluation by calling Fireworks directly (skip LiteLLM router)
local-eval-fireworks-only:
	@test -n "$$FIREWORKS_API_KEY" || (echo "FIREWORKS_API_KEY not set" && exit 1)
	uv run pytest eval_protocol/quickstart/llm_judge_langfuse_fireworks_only.py -k test_llm_judge_fireworks_only -q

# One-shot: assumes Langfuse is already up externally and LiteLLM already running in another shell
local-quick-run: local-seed-langfuse local-adapter-smoke local-eval
	@echo "Done. Check Langfuse UI for scores."


clean:
	rm -rf build/ dist/ *.egg-info/

pre-commit:
	pre-commit run --all-files

build: clean
	python -m build

dist: build

upload:
	twine upload dist/*

test:
	pytest

lint:
	flake8 $(PYTHON_DIRS)

typecheck:
	mypy $(PYTHON_DIRS)

format:
	black $(PYTHON_DIRS)

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
	@echo "  lint          - Run flake8 linter"
	@echo "  typecheck     - Run mypy type checker"
	@echo "  format        - Run black code formatter"
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
