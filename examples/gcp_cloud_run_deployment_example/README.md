# GCP Cloud Run Deployment Example for `eval-protocol`

This example demonstrates how to deploy a simple reward function (`dummy_rewards.py`) to Google Cloud Run using the `eval-protocol` CLI.

## Files

-   `dummy_rewards.py`: Contains a basic `hello_world_reward` function.
-   `rewardkit.example.yaml`: An example configuration file for `eval-protocol`. You should copy this to the root of your project (where you'll run `eval-protocol` commands) as `rewardkit.yaml` and customize it.

## Prerequisites

1.  **Google Cloud Platform (GCP) Account:** You need an active GCP account with billing enabled.
2.  **`gcloud` CLI:** Ensure the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) is installed and authenticated (`gcloud auth login`, `gcloud auth application-default login`).
3.  **APIs Enabled:** Make sure the following APIs are enabled in your GCP project:
    *   Cloud Build API
    *   Artifact Registry API
    *   Cloud Run Admin API
    *   Secret Manager API
4.  **Permissions:** The authenticated user/service account for `gcloud` needs sufficient permissions to manage these services (e.g., roles like "Cloud Build Editor", "Artifact Registry Administrator", "Cloud Run Admin", "Secret Manager Admin").
5.  **`eval-protocol` installed:** Ensure `eval-protocol` is installed in your Python environment.

## Setup

1.  **Navigate to this Example Directory:**
    Open your terminal and change to this directory:
    ```bash
    cd path/to/eval-protocol/examples/gcp_cloud_run_deployment_example
    ```

2.  **Configure `rewardkit.yaml`:**
    *   In this directory (`examples/gcp_cloud_run_deployment_example/`), copy `rewardkit.example.yaml` to a new file named `rewardkit.yaml`.
        ```bash
        cp rewardkit.example.yaml rewardkit.yaml
        ```
    *   Open the new `rewardkit.yaml` file and **replace placeholders** like `your-gcp-project-id-here` with your actual GCP Project ID and desired region. For example:
        ```yaml
        # rewardkit.yaml (example content after customization)
        gcp_cloud_run:
          project_id: "my-actual-gcp-project-123"
          region: "us-west1"
          # artifact_registry_repository: "my-custom-eval-repo" # Optional
          # default_auth_mode: "api-key" # Optional, defaults to api-key
        evaluator_endpoint_keys: {}
        ```
    *   When you run `eval-protocol` commands from this directory, it will automatically find and use this `rewardkit.yaml`.

    **Note:** If you prefer not to use a `rewardkit.yaml` file here, you can skip step 2 and instead provide all necessary GCP parameters (`--gcp-project`, `--gcp-region`) directly via CLI arguments in the deployment command below.

## Deployment Command

Ensure you are in the `examples/gcp_cloud_run_deployment_example/` directory.

Run the following command:

```bash
eval-protocol deploy \
    --id my-dummy-gcp-evaluator \
    --target gcp-cloud-run \
    --function-ref dummy_rewards.hello_world_reward \
    --gcp-auth-mode api-key \
    --verbose
    # --force # Add this flag if you need to overwrite an existing evaluator with the same ID
    # If you didn't configure rewardkit.yaml, add:
    # --gcp-project YOUR_PROJECT_ID --gcp-region YOUR_REGION
```

**Explanation:**
-   `--id my-dummy-gcp-evaluator`: A unique ID for your evaluator.
-   `--target gcp-cloud-run`: Specifies deployment to GCP Cloud Run.
-   `--function-ref dummy_rewards.hello_world_reward`: The Python import path to your reward function. Since `dummy_rewards.py` is in the current directory (when running from here), this simple reference should work.
-   `--gcp-auth-mode api-key`: Deploys the service with API key authentication (this is the default if not specified). `eval-protocol` will generate a key, store it in GCP Secret Manager, and configure the service. The key will also be saved to your local `rewardkit.yaml` (in this directory) under `evaluator_endpoint_keys`.
-   `--verbose`: Shows detailed output, including `gcloud` commands being executed.
-   `--force`: (Optional) If an evaluator with the same `--id` already exists, adding `--force` will delete the existing one before creating the new one. Without it, you'll get a conflict error if the ID is taken.

## Expected Outcome

If successful, `eval-protocol` will:
1.  Create an Artifact Registry repository (if it doesn't exist, default: `eval-protocol-evaluators`).
2.  Build a Docker container with your reward function and push it to Artifact Registry.
3.  Create a GCP Secret to store the generated API key for your service.
4.  Deploy the container to Cloud Run, configured with the API key from Secret Manager.
5.  Register the deployed Cloud Run service URL as a remote evaluator with the Fireworks AI platform.

You will see the Cloud Run service URL and the API key (if newly generated) in the output. The API key will also be printed by the `eval_protocol.generic_server.py` when it starts up inside the Cloud Run container if `RK_ENDPOINT_API_KEY` is set.

## Testing the Deployed Endpoint

You can test the deployed endpoint using `curl` or `eval-protocol preview --remote-url <your-cloud-run-url>`.
If using `curl` with API key auth:
```bash
# Get your API key from rewardkit.yaml or the deploy command output
API_KEY="your_generated_api_key"
SERVICE_URL="your_cloud_run_service_url"

curl -X POST "$SERVICE_URL/evaluate" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $API_KEY" \
     -d '{
           "messages": [{"role": "user", "content": "Test"}],
           "kwargs": {}
         }'
