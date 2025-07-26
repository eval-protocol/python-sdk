# Plan: Remote Evaluation with Secrets

This document outlines plans related to remote evaluation capabilities in `eval-protocol`.
It first covers the setup for a local demonstration using automated tunneling (Serveo.net), and then details a future vision for first-class support of self-hosted remote evaluators on cloud platforms like GCP and AWS.

## Part 1: Local Remote Evaluation Demo (Serveo.net Tunneling) - Status: DONE

This section details the plan and implementation of a demonstration for remote evaluation capabilities using a local mock server exposed via Serveo.net.

### 1.1. Objectives - Status: DONE
(Content as before)
*   Demonstrate that evaluation functions can call arbitrary remote URLs.
*   Showcase secure secret management for API keys using environment variables (simulating `fireworks secret` behavior).
*   Illustrate a less secure method (hardcoding) for comparison.
*   Provide a utility for API key generation.
*   Adapt an existing simple reward function to fit this scenario for local testing.
*   Enable a one-command execution (`make demo-remote-eval`) that sets up all necessary components, including the public tunnel, using Serveo.net.

### 1.2. Components - Status: DONE
(Content as before)
#### 1.2.1. Mock API Service (FastAPI) - Status: DONE
*   Simulates an external service requiring API key authentication.
*   Implemented in `examples/remote_eval_demo/mock_api_service.py`.
*   Reads expected API key and port from `EXPECTED_API_KEY` and `PORT` environment variables.

#### 1.2.2. Tunneling Component (Serveo.net) - Status: DONE
*   Exposes the local FastAPI service using Serveo.net via SSH.
*   Automated by `run_demo.py` using `development/utils/subprocess_manager.py`.

#### 1.2.3. Evaluation Functions - Status: DONE
*   `examples/remote_eval_demo/rewards/remote_validator_reward_hardcoded.py`
*   `examples/remote_eval_demo/rewards/remote_validator_reward_secure.py`
*   These functions call the tunneled FastAPI service and demonstrate different secret handling strategies. URL parameters generalized to `target_service_url`. Pydantic validation for `EvaluateResult.metrics` fixed.

#### 1.2.4. Secret Management Strategies - Status: DONE
*   Strategy A (Hardcoded Secret): Demonstrated in `remote_validator_reward_hardcoded.py`.
*   Strategy B (Environment Variable): Demonstrated in `remote_validator_reward_secure.py` (reads from `MOCK_SERVICE_API_KEY` env var) and in `mock_api_service.py` (reads `EXPECTED_API_KEY` env var).

### 1.3. Detailed Implementation Steps - Status: DONE
(Content as before)
*   **API Key Generation Utility (`development/utils/generate_api_key.py`):** DONE. Prints raw key.
*   **FastAPI Mock Service (`examples/remote_eval_demo/mock_api_service.py`):** DONE. Reads API key and port from env vars.
*   **Tunneling Automation (`development/utils/subprocess_manager.py`):** DONE. `start_serveo_and_get_url` implemented. Ngrok functions deprecated. `start_process` correctly handles custom `env` for subprocesses.
*   **Reward Functions:** DONE. Updated for generic URL params and Pydantic compliance.
*   **Demo Script (`examples/remote_eval_demo/run_demo.py`):** DONE. Fully automates API key generation, mock service startup (with correct env vars), Serveo tunnel, and test execution.
*   **Makefile Target (`make demo-remote-eval`):** DONE.
*   **Documentation (`examples/remote_eval_demo/README.md`):** DONE. Overhauled for Serveo.

### 1.4. Files Created/Modified - Status: DONE
(Content as before)
All relevant files for the Serveo.net demo have been created and modified as per the original plan and subsequent debugging.

### 1.5. Considerations (General) - Status: ADDRESSED
(Content as before)
Error handling, idempotency, clarity, and version compatibility were considered and addressed during implementation.

### 1.6. Automating Tunneling with Serveo.net - Status: DONE
(Content as before)
The implementation using Serveo.net for the one-command demo (`make demo-remote-eval`) is complete.

---

## Part 2: Future Vision - First-Class Self-Hosted Remote Evaluators (GCP/AWS)

This section outlines a plan to significantly enhance `eval-protocol` to provide a seamless, "one-command" experience for users to deploy their Python reward functions to their own cloud infrastructure (initially GCP Cloud Run and AWS Lambda) and register these as remote evaluators with the Fireworks AI platform. The core principle is **ease of use**: the user writes only their reward function logic, and `eval-protocol` handles the complexities of packaging, cloud deployment, and secure secret management.

### 2.1. Objectives
(Content as before)
*   Enable users to deploy reward functions to their own GCP Cloud Run or AWS Lambda environments with a single `eval-protocol` CLI command.
*   Abstract away most cloud-provider-specific complexities from the user.
*   Provide a "zero-wrapper" experience: users only write their Python reward function module.
*   Integrate secure secret management for secrets used *by* the reward function (e.g., API keys for third-party services), leveraging cloud provider secret managers (GCP Secret Manager, AWS Secrets Manager).
*   Enhance endpoint security for these self-hosted evaluators (e.g., API key, IAM, mTLS).
*   Allow easy registration of these self-hosted evaluators with the Fireworks AI platform.
*   Support previewing against these self-hosted evaluators.

### 2.2. Target CLI User Experience
(Content as before)
The envisioned CLI commands would simplify deployment and previewing:

*   **Deploy to Cloud:**
    *   `eval-protocol deploy <function_ref> --target gcp-cloud-run [--evaluator-id <id>] [--project <gcp_project>] [--region <gcp_region>] [--service-name <name>] [--auth <api-key|iam|mtls-client-auth>] [--secrets ENV_VAR_NAME=provider_secret_id,...]`
    *   `eval-protocol deploy <function_ref> --target aws-lambda [--evaluator-id <id>] [--region <aws_region>] [--function-name <name>] [--auth <api-key|iam|mtls-client-auth>] [--secrets ENV_VAR_NAME=provider_secret_id,...]`
    *   `<function_ref>` is a Python import string like `my_module.my_reward_func`.
    *   `--secrets` maps environment variables for the reward function to IDs/ARNs in the cloud provider's secret manager.
    *   `--auth` specifies the authentication method for the deployed endpoint.

*   **Preview against Self-Hosted:**
    *   `eval-protocol preview <function_ref_or_id> --target gcp-cloud-run [--service-name <name>] --samples <file>` (if `eval-protocol` can discover the URL from cloud provider based on name/config).
    *   `eval-protocol preview --remote-url <user_provided_cloud_url> --samples <file>` (for any existing URL, including self-hosted).

*   **Local Serving (for development/testing, leveraging the same internal server):**
    *   `eval-protocol deploy <function_ref> --local-serve [--tunnel auto] --id <evaluator_id>`
    *   `eval-protocol preview <function_ref> --local-serve`

### 2.3. Core `eval-protocol` Enhancements Required (Phase A Status Update)

1.  **Internal Generic Reward Function Server:** - **Status: DONE**
    *   A built-in HTTP server (FastAPI-based) within `eval-protocol` (`eval_protocol/generic_server.py`) capable of dynamically loading and serving any user-provided Python reward function.
    *   Exposes a standardized `/evaluate` endpoint and a `/health` endpoint.
    *   Handles request/response serialization for `EvaluateResult`.
    *   Unit and integration tests created and passing (`tests/test_generic_server.py`).

2.  **Project Configuration File (`rewardkit.yaml`):** - **Status: DONE**
    *   A local file (`rewardkit.yaml`) can be used to store stable settings.
    *   Implemented loading logic in `eval_protocol/config.py` with Pydantic models.
    *   Unit tests created and passing (`tests/test_config.py`).
        ```yaml
        # Example rewardkit.yaml structure
        default_deployment_target: gcp-cloud-run

        gcp_cloud_run:
          project_id: "my-gcp-project"
          region: "us-central1"
          service_name_template: "rewardeval-{evaluator_id}"
          default_auth_mode: "api-key"
          secrets:
            MY_FUNCTION_API_KEY: "projects/my-gcp-project/secrets/my-fn-api-key/versions/latest"

        aws_lambda:
          region: "us-east-1"
          function_name_template: "rewardeval-{evaluator_id}"
          default_auth_mode: "api-key"
          secrets:
            MY_FUNCTION_API_KEY: "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-fn-api-key-xxxxxx"

        evaluator_endpoint_keys:
          my_gcp_eval_id: "generated_secure_key_for_endpoint"
        ```

3.  **Enhanced `eval-protocol deploy` Command (for `--remote-url`):** - **Status: DONE**
    *   Supports `--remote-url <url>` for registering an existing URL with the Fireworks AI platform.
    *   CLI argument parsing and command logic in `eval_protocol/cli.py` and `eval_protocol/cli_commands/deploy.py` are implemented and tested.
    *   The Fireworks AI platform API call is handled by `eval_protocol.evaluation.create_evaluation`, which deploys a Python shim that proxies to the specified remote URL. This is the current accepted mechanism.

4.  **Enhanced `eval-protocol preview` Command (for `--remote-url`):** - **Status: DONE**
    *   Supports `--remote-url <url>` to preview against any remote evaluator endpoint.
    *   CLI argument parsing and command logic in `eval_protocol/cli.py` and `eval_protocol/cli_commands/preview.py` are implemented and tested.
    *   Uses the robust sample loaders from `common.py`.

5.  **Robust Sample Loading Utilities:** - **Status: DONE**
    *   Implemented `load_samples_from_file` and `load_samples_from_huggingface` in `eval_protocol/cli_commands/common.py`.
    *   Includes validation, error handling, and logging.
    *   Comprehensive unit tests created and passing (`tests/cli_commands/test_common.py`).

6.  **Test Suite Health:** - **Status: IMPROVED**
    *   Addressed all reported test failures and hangs across `tests/test_generic_server.py`, `tests/test_config.py`, `tests/test_cli_args.py`, `tests/cli_commands/test_preview_cmd.py`, `tests/cli_commands/test_deploy_cmd.py`, `tests/test_cli.py`, `tests/test_evaluation.py`, and `tests/test_evaluation_integration.py`.

**Phase A Completion Note:**
*   The previously stated "Next Immediate Step" regarding the API call for `eval-protocol deploy --remote-url` is now considered complete. The existing functionality in `eval_protocol.evaluation.create_evaluation` (which deploys a Python shim to proxy to the remote URL) serves this purpose. The concept of a separate, direct API endpoint for remote URL registration (previously explored in `eval_protocol.platform_api.py`) has been deprecated as no such backend API exists.

With this, Phase A is considered complete.

---
The following sections describe future work.
---

7.  **Platform-Specific Packaging Logic:** (Future Work for Phase B/C)
    *   **For GCP Cloud Run:**
        *   Generate a `Dockerfile`.
        *   Orchestrate `docker build` and `docker push`.
    *   **For AWS Lambda:**
        *   Create a Lambda deployment package (.zip).

8.  **Cloud Provider CLI Orchestration:** (Future Work for Phase B/C)
    *   Invoke `gcloud` and `aws` CLI commands.

### 2.4. Enhanced Security Considerations (Future Work for Phase B/C/D)
(Content as before)
1.  **Protecting the Deployed Reward Function Endpoint:**
    *   **Default (`--auth api-key`):**
    *   **IAM (`--auth iam`):**
    *   **mTLS - Client Certificate Validation (`--auth mtls-client-auth`):**

2.  **Managing Secrets Used *by* the Reward Function:**
    *   The `--secrets ENV_VAR_NAME=provider_secret_id` flag.

### 2.5. Phased Implementation Approach (Updated Summary)

1.  **Phase A: Core Framework (Largely Complete):**
    *   **DONE:** Internal Generic Reward Function Server.
    *   **DONE:** `rewardkit.yaml` basic structure and loading.
    *   **DONE:** Enhance `eval-protocol deploy` to support `--remote-url <url>`.
        *   CLI and argument handling complete.
        *   The platform API call is handled by `eval_protocol.evaluation.create_evaluation` (deploys a Python shim proxy). This fulfills the requirement.
    *   **DONE:** Enhance `eval-protocol preview` to support `--remote-url <url>`.
    *   **DONE:** Robust sample loading utilities in `eval_protocol/cli_commands/common.py`.
    *   **DONE:** Resolved test failures across multiple suites.

2.  **Phase B: GCP Cloud Run Integration (Future Work):**
    *   Implement `eval-protocol deploy ... --target gcp-cloud-run`.
    *   Dockerfile generation and `gcloud` orchestration.
    *   Support for `--auth api-key`.
    *   Support for `--secrets` mapping from GCP Secret Manager.

3.  **Phase C: AWS Lambda Integration (Future Work):**
    *   Implement `eval-protocol deploy ... --target aws-lambda`.
    *   Lambda packaging and `aws` CLI orchestration.
    *   Support for `--auth api-key`.
    *   Support for `--secrets` mapping from AWS Secrets Manager.

4.  **Phase D: Advanced Authentication Modes (Future Work):**
    *   Implement `--auth iam` for GCP and AWS.
    *   Implement `--auth mtls-client-auth` for GCP and AWS.

5.  **Phase E: Local Secret Store (`eval-protocol secret add --project-local ...`) (Future Work):**
    *   If relying solely on environment variables for local development proves insufficient, implement a local, project-specific secret store.

This updated plan aims for a highly user-friendly "one-command" experience for self-hosting reward functions, while also providing robust security options and abstracting cloud-specific details. The immediate next step is to complete the backend integration for `eval-protocol deploy --remote-url`.
