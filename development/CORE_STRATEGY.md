# Reward-Protocol: Core Strategy & Roadmap

## Punch Line

**Reward-Protocol: Author, reproduce, and evaluate reward functions seamlessly on Fireworks, TRL, and your own infrastructure.**

## Core Tenets

These tenets guide the development and evolution of Reward-Protocol:

*   **T1: Empower Effortless Reward Function Authoring & Unwavering Reproducibility.**
    *   Reward-Protocol must provide an intuitive and simple interface for defining reward functions. The process from idea to a functioning reward function should be as frictionless as possible. Crucially, all aspects of evaluation, including data, verifier setups, and execution environments, must be easily and reliably reproducible to ensure consistent results and facilitate debugging and iteration.

*   **T2: Champion Broad Integration & Principled Extensibility.**
    *   Reward-Protocol aims for deep and seamless integration with key platforms like Fireworks AI and popular RL frameworks such as TRL. Beyond these, it must be architected to support diverse execution infrastructures (local, managed cloud services like GCP/Lambda, or user-owned servers) and be extensible for new reward types, data sources, and evaluation paradigms.

*   **T3: Deliver Robust Evaluation Capabilities & Insightful Debugging.**
    *   The library must offer powerful and flexible tools for evaluating reward functions. This includes not just scoring, but also rich metric collection. Integration with UIs like Fireworks for detailed review, comparison, and debugging of evaluators is paramount. Reward-Protocol should facilitate experimentation, allowing users to easily test different models, parameters (e.g., pass @ k), and configurations.

*   **T4: Uphold Enterprise-Grade Security & Trust.**
    *   As Reward-Protocol handles potentially sensitive models and data, and integrates with various services, security is non-negotiable. It must support robust authentication mechanisms (API keys, IAM roles, mTLS) to protect developer workloads and ensure that integrations do not compromise the security posture of user environments.

*   **T5: Foster Learning & Adoption through Rich Examples & Crystal-Clear Documentation.**
    *   To be truly useful, Reward-Protocol needs comprehensive, practical examples that showcase its capabilities across various use cases. These examples should be paired with well-defined datasets. All documentation, from high-level guides to API references, must be clear, accurate, up-to-date, and focused on enabling users to achieve their goals quickly and effectively.

## Phased Implementation Roadmap

This roadmap outlines the development priorities. Tasks within each phase can be parallelized where appropriate.

*(Original Task 1: Review and Consolidate Existing Development Documents - This meta-task is now considered complete as all identified dev docs have been processed or marked for processing. Specific plans from `development/readiness/*.md` are integrated into example creation tasks below.)*

---
**Phase 1: Foundational Stability, Developer Experience (DX), and Critical Documentation**
*(Focus: Stabilize core APIs, improve code quality infrastructure, fix critical documentation, and enhance basic usability.)*

*   **P1.1: [CRITICAL] Improve Agent Multi-Step Rollout Implementation:**
    *   **Detailed Plan:** See [Multi-Step RL Enhancement Plan Overview](./notes/multi_step_rl_enhancement_plan_overview.md)
    *   [ ] Review existing agent multi-step rollout capabilities (as per Phase 0 in detailed plan).
    *   [ ] Implement Phase 1: Foundational RL Enablement (Tasks 1.1 - 1.5).
    *   [ ] Implement Phase 2: Advanced RL Algorithms and Observability (Tasks 2.1 - 2.4).
    *   [ ] Implement Phase 3: Scalability and Productionization (Tasks 3.1 - 3.2).
    *   [ ] Update relevant documentation and examples throughout the implementation.
*   **P1.2: [CRITICAL] Complete API & Data Format Refactor:**
    *   [ ] **Core Reward Functions:** Finish refactoring all remaining functions in `reward_kit/rewards/` (e.g., `multiple_choice_math_reward.py`, `reasoning_steps.py`, `accuracy_length.py`) and their tests to align with the new `messages`/`ground_truth` paradigm and Pydantic type hinting.
    *   [ ] **Example Scripts:** Update all existing example scripts in `examples/` to use the new API and data formats.
    *   [ ] **TRL Adapter:** Update `RewardFunction.get_trl_adapter` in `reward_kit/reward_function.py`.
    *   [ ] **`RewardFunction` Class:** Review and align `reward_kit/reward_function.py` (especially `__call__` and `func_path` handling).
*   **P1.4: Code Quality & Tooling Setup:**
    *   [ ] **Configuration Standardization:** Standardize `flake8`, `black`, `mypy` configurations across `.pre-commit-config.yaml`, `.flake8`, `mypy.ini`, `Makefile`, `pyproject.toml` (if adopted).
    *   [ ] **MyPy Strictness:** Systematically review and reduce globally disabled MyPy error codes.
    *   [ ] **Makefile & Pre-commit Consistency:** Ensure Makefile targets and pre-commit hooks are consistent.
    *   [ ] **Docstring Checking:** Enable and enforce docstring checking.
*   **P1.5: Authentication System Refactor:**
    *   [ ] **Centralize Logic:** Create `reward_kit/auth.py`.
    *   [ ] **Configuration Methods:** Support environment variables and `auth.ini`.
    *   [ ] **Documentation:** Clearly document `FIREWORKS_ACCOUNT_ID` usage and auth setup.
    *   [ ] **Codebase Update:** Refactor `reward_kit/evaluation.py` etc. to use new auth module.
*   **P1.6: Build, Packaging, and Basic CI:**
    *   [ ] **`setup.py` Review:** Evaluate `openai` pinning, clean `extras_require`, populate `long_description`.
    *   [ ] **Basic CI Pipeline (GitHub Actions):** Setup CI for linting, formatting checks, and running tests on main Python version.
    *   [ ] **`CHANGELOG.md`:** Backfill missing versions (`0.2.5` to `0.2.11`) and establish update process.

---
**Phase 2: Core Feature Enhancements, Dataset Management & New Example Suites**
*(Focus: Develop core utilities, key integrations, robust dataset handling, and new comprehensive examples that showcase the power of Reward-Protocol.)*

*   **P2.1: Hydra-based Dataset Management Refactor:**
    *   [ ] **Core Dataset Module:** Create `reward_kit/datasets/loader.py` (or similar) for centralized dataset loading/processing logic driven by Hydra.
    *   [ ] **Hydra Schemas:** Define structured Hydra configuration schemas for datasets (e.g., in `conf/dataset/base_dataset.yaml`, `conf/dataset/gsm8k.yaml`).
    *   [ ] **Refactor `convert_dataset.py`:** Further refine `examples/math_example/convert_dataset.py` to use the new global Hydra dataset schemas.
    *   [ ] **Refactor Example Scripts:** Update other example scripts (`local_eval.py`, TRL examples, etc.) to use Hydra for dataset configuration, referencing global dataset configs.
    *   [ ] **Documentation:** Update `CONTRIBUTING.md` and example READMEs to explain the new dataset configuration system.
*   **P2.2: Core Utility Development:**
    *   [ ] Develop robust JSONL and conversational dataset loading utilities.
    *   [ ] Create a client-side utility for Fireworks AI chat completions (sync/async).
    *   [ ] Implement an API mocking utility for testing.
    *   [ ] Develop math-specific string processing utilities (if still needed).
*   **P2.3: TRL Integration Development:**
    *   [ ] Create a generic adapter for using `reward-protocol` reward functions with TRL.
*   **P2.4: Comprehensive Coding Example Creation:**
    *   [ ] Curate/generate `examples/coding_example/dataset.jsonl`.
    *   [ ] Create `local_eval.py`, `fireworks_preview.py`, `fireworks_regenerate.py`, `trl_grpo_integration.py` for coding.
    *   [ ] Add E2E tests for this example.
*   **P2.5: Composite Math & Coding Example Creation:**
    *   [ ] Define composite problem structure and dataset.
    *   [ ] Develop composite reward logic.
    *   [ ] Create `local_eval.py`, `fireworks_preview.py`, `fireworks_regenerate.py`, `trl_grpo_integration.py` for composite.
    *   [ ] Add E2E tests for this example.
*   **P2.6: Example Standardization & Refinements:**
    *   [ ] Define consistent configuration management for examples.
    *   [ ] Develop templates for common example scripts.
    *   [ ] Standardize data preparation steps for examples.
    *   [ ] Standardize internal mocking patterns in examples.
    *   [ ] Refactor `examples/math_example/` for shared logic.
    *   [ ] Align specific TRL examples (e.g., math) with generic TRL examples.
*   **P2.7: SDK - Kick Off Evaluation Runs:**
    *   [ ] Develop SDK capabilities for initiating evaluation runs programmatically.

---
**Phase 3: Advanced Features, Operational Excellence & Broader Ecosystem**
*(Focus: Implement advanced features like self-hosted evaluators, mature CI/CD, and further improve documentation and developer operations.)*

*   **P3.1: SDK - Kick Off RL Jobs:**
    *   [ ] Develop SDK capabilities for initiating RL jobs (potentially leveraging TRL adapter).
*   **P3.2: Advanced Authentication for Self-Hosted Evaluators:**
    *   [ ] Implement IAM and mTLS options for self-hosted evaluators (requires self-hosted infrastructure from P3.3).
*   **P3.3: Future Vision - Self-Hosted Remote Evaluators:**
    *   **GCP Cloud Run Integration:** Implement `reward-protocol deploy ... --target gcp-cloud-run`, Dockerfile generation, `gcloud` orchestration, API key auth, GCP Secret Manager integration.
    *   **AWS Lambda Integration:** Implement `reward-protocol deploy ... --target aws-lambda`, Lambda packaging, `aws` CLI orchestration, API key auth, AWS Secrets Manager integration.
    *   **(Future) Local Secret Store (Optional):** Implement if needed.
*   **P3.4: Advanced CI/CD & Release Process:**
    *   [ ] Full CI pipeline: multi-Python version testing, build sdist/wheel, docs build check, coverage reporting.
    *   [ ] Version Management: Formalize SemVer.
    *   [ ] Release Automation: Auto-create GitHub releases, publish to PyPI on tags.
*   **P3.5: GitHub Operations & Security:**
    *   [ ] Create Issue and PR templates.
    *   [ ] Enable and configure GitHub Dependabot.
*   **P3.6: Comprehensive Documentation Overhaul:**
    *   [ ] Complete review and update of all `docs/` content.
    *   [ ] Ensure clarity on reproducibility, platform execution, RL integration.
    *   [ ] Polish `README.md` and `CONTRIBUTING.md` further.
    *   [ ] Address all remaining documentation polish items (links, metric folder clarity, editorial comments, etc.).
*   **P3.7: Ongoing Dataset & Example Maintenance/Enhancement:**
    *   [ ] Continue to refine `CODING_DATASET.jsonl`.
    *   [ ] Improve pass rates for Math examples (GSM8K, OpenR1).
    *   [ ] Develop other new examples as identified.
*   **P3.8: Advanced Testing:**
    *   [ ] Strengthen TRL E2E test verification (stdout parsing).
    *   [ ] Consider and plan "out-of-the-box" smoke test suite.

---
**Ongoing / Cross-Cutting Concerns:**
*   Relocating general-purpose development utilities.
*   LICENSE & Legal: `NOTICE` file, source file headers.
*   API Spec Reference in docs.
*   Repository Hygiene.
*   Community Engagement Plan.
*   DeepCoder-Style Reward Experiment continuation.
*   (Future Consideration) `pyproject.toml` migration.
