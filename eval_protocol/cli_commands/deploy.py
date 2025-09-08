"""
CLI command for creating and deploying an evaluator,
or registering a pre-deployed remote evaluator.
"""

import importlib  # For dynamically importing modules
import json
import os  # For os.path.join, os.makedirs, os.getcwd (already imported but good to be explicit if used extensively)
import secrets  # For API key generation (already imported but good to be explicit)
import sys  # For sys.executable
import time  # For sleep
from pathlib import Path  # For path operations
from typing import Any, Dict

import yaml  # For saving config if save_config helper doesn't exist

# TODO: Consider moving subprocess_manager functions to a more central location if used by core CLI
try:
    # Import functions with explicit names to match expected signatures
    from development.utils.subprocess_manager import (
        start_ngrok_and_get_url as _start_ngrok_and_get_url,
        start_process as _start_process,
        start_serveo_and_get_url as _start_serveo_and_get_url,
        stop_process as _stop_process,
    )
except ImportError:
    # Fallback implementations when development module is not available
    import signal
    import socket
    import subprocess

    def _fallback_start_process(command, log_path, env=None):
        """Fallback process starter."""
        try:
            with open(log_path, "w") as log_file:
                process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
                return process
        except Exception as e:
            print(f"Error starting process: {e}")
            return None

    def _fallback_stop_process(pid):
        """Fallback process stopper."""
        try:
            import os

            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    def _fallback_start_serveo_and_get_url(local_port, log_path):
        """Fallback serveo tunnel - returns None to indicate unavailable."""
        print("Serveo tunneling not available - development module not found")
        return None, None

    def _fallback_start_ngrok_and_get_url(local_port, log_path):
        """Fallback ngrok tunnel - returns None to indicate unavailable."""
        print("ngrok tunneling not available - development module not found")
        return None, None

    # Expose unified names using fallbacks
    start_process = _fallback_start_process
    stop_process = _fallback_stop_process
    start_serveo_and_get_url = _fallback_start_serveo_and_get_url
    start_ngrok_and_get_url = _fallback_start_ngrok_and_get_url
else:
    # Wrap imported helpers to present consistent, simple signatures used below
    def start_process(command, log_path, env=None):
        return _start_process(command=command, log_file_path=log_path, env=env)

    def stop_process(pid):
        return _stop_process(pid)

    def start_serveo_and_get_url(local_port, log_path):
        return _start_serveo_and_get_url(local_port=local_port, log_file_path=log_path)

    def start_ngrok_and_get_url(local_port, log_path):
        return _start_ngrok_and_get_url(local_port=local_port, ngrok_log_file=log_path)


from eval_protocol.auth import get_fireworks_account_id
from eval_protocol.config import (
    GCPCloudRunConfig,
    RewardKitConfig,
    _config_file_path as global_loaded_config_path,
    get_config,
)
from eval_protocol.evaluation import create_evaluation
from eval_protocol.gcp_tools import (
    build_and_push_docker_image,
    deploy_to_cloud_run,
    ensure_artifact_registry_repo_exists,
    ensure_gcp_secret,
)
from eval_protocol.packaging import generate_dockerfile_content
from eval_protocol.platform_api import (  # For catching errors from create_evaluation
    PlatformAPIError,
    create_or_update_fireworks_secret,
)

from .common import check_environment


def _establish_local_server_and_tunnel(args):
    """
    Handles starting the local generic server and establishing a public tunnel
    using Serveo, with a fallback to ngrok.
    Returns: (public_url, tunnel_provider_name, local_server_pid, tunnel_process_pid)
             Returns (None, None, server_pid_or_None, None) if tunneling fails.
    """
    if not args.function_ref:
        print("Error: --function-ref is required for local-serve target.")
        return None, None, None, None

    evaluator_id = args.id
    function_ref = args.function_ref
    local_server_port = args.local_port

    log_dir = os.path.join(os.getcwd(), "logs", "eval-protocol-local")
    os.makedirs(log_dir, exist_ok=True)
    generic_server_log_path = os.path.join(log_dir, f"generic_server_{evaluator_id}.log")

    server_env = None  # Run local server without API key protection
    print(f"Note: Local server for '{evaluator_id}' will run without API key protection.")

    print(f"Starting local reward function server for '{function_ref}' on port {local_server_port}...")
    server_command = [
        sys.executable,
        "-m",
        "eval_protocol.generic_server",
        function_ref,
        "--port",
        str(local_server_port),
    ]

    local_server_process = start_process(server_command, generic_server_log_path, env=server_env)

    if not local_server_process or local_server_process.poll() is not None:
        print(f"Error: Failed to start local generic server. Check log: {generic_server_log_path}")
        return None, None, None, None  # No server, no tunnel

    local_server_pid = local_server_process.pid
    print(f"Local server started (PID: {local_server_pid}). Log: {generic_server_log_path}")
    print("Waiting for server to initialize...")
    time.sleep(5)

    # Attempt Serveo first
    print(f"Attempting Serveo tunnel for local port {local_server_port}...")
    serveo_log_path = os.path.join(log_dir, f"serveo_{evaluator_id}.log")
    serveo_tunnel_process, serveo_url = start_serveo_and_get_url(local_server_port, serveo_log_path)

    if serveo_url and serveo_tunnel_process:
        print(f"Serveo tunnel established: {serveo_url} (PID: {serveo_tunnel_process.pid}). Log: {serveo_log_path}")
        return serveo_url, "serveo", local_server_pid, serveo_tunnel_process.pid
    else:
        print(f"Serveo tunnel failed. Check log: {serveo_log_path}")
        print("Attempting fallback to ngrok...")

        ngrok_log_path = os.path.join(log_dir, f"ngrok_{evaluator_id}.log")
        # Assuming ngrok authtoken is pre-configured by the user or via NGROK_AUTHTOKEN env var
        ngrok_tunnel_process, ngrok_url = start_ngrok_and_get_url(local_server_port, ngrok_log_path)

        if ngrok_url and ngrok_tunnel_process:
            print(f"ngrok tunnel established: {ngrok_url} (PID: {ngrok_tunnel_process.pid}). Log: {ngrok_log_path}")
            return ngrok_url, "ngrok", local_server_pid, ngrok_tunnel_process.pid
        else:
            print(f"ngrok tunnel also failed. Check log: {ngrok_log_path}")
            # Both failed, stop the local server we started
            if local_server_pid:
                stop_process(local_server_pid)
            return (
                None,
                None,
                local_server_pid,
                None,
            )  # URL, provider, server_pid, tunnel_pid


def _deploy_to_gcp_cloud_run(args, current_config, gcp_config_from_yaml):
    """Handles the logic for --target gcp-cloud-run up to service deployment."""
    print(f"Starting GCP Cloud Run deployment for evaluator '{args.id}'...")

    # Resolve function_ref (must be from CLI for GCP)
    if not args.function_ref:  # This check is also in main, but good for helper too
        print("Error: --function-ref is required for GCP Cloud Run deployment.")
        return None

    # Dynamically import the reward function to get its requirements
    inline_requirements_content = None
    try:
        module_name, func_name = args.function_ref.rsplit(".", 1)
        module = importlib.import_module(module_name)
        reward_func = getattr(module, func_name)
        if hasattr(reward_func, "_reward_function_requirements"):
            inline_requirements_content = reward_func._reward_function_requirements
            if inline_requirements_content:
                print(f"Found inline requirements for {args.function_ref}")
    except Exception as e:
        print(f"Warning: Could not load reward function {args.function_ref} to check for inline requirements: {e}")
        # Continue without inline requirements if loading fails

    # Resolve GCP project_id
    gcp_project_id = args.gcp_project
    if not gcp_project_id and gcp_config_from_yaml:
        gcp_project_id = gcp_config_from_yaml.project_id
    if not gcp_project_id:
        print("Error: GCP Project ID must be provided via --gcp-project argument or in rewardkit.yaml.")
        return None

    # Resolve GCP region
    gcp_region = args.gcp_region
    if not gcp_region and gcp_config_from_yaml:
        gcp_region = gcp_config_from_yaml.region
    if not gcp_region:
        print("Error: GCP Region must be provided via --gcp-region argument or in rewardkit.yaml.")
        return None

    # Resolve GCP AR repo name
    gcp_ar_repo_name = args.gcp_ar_repo
    if not gcp_ar_repo_name and gcp_config_from_yaml:
        gcp_ar_repo_name = gcp_config_from_yaml.artifact_registry_repository
    if not gcp_ar_repo_name:
        gcp_ar_repo_name = "eval-protocol-evaluators"

    print(f"Using GCP Project: {gcp_project_id}, Region: {gcp_region}, AR Repo: {gcp_ar_repo_name}")

    if not ensure_artifact_registry_repo_exists(
        project_id=gcp_project_id, region=gcp_region, repo_name=gcp_ar_repo_name
    ):
        print(f"Failed to ensure Artifact Registry repository '{gcp_ar_repo_name}' exists. Aborting.")
        return None

    dockerfile_content = generate_dockerfile_content(
        function_ref=args.function_ref,
        python_version=(
            f"{args.runtime[6]}.{args.runtime[7:]}"
            if args.runtime.startswith("python") and len(args.runtime) > 7
            else args.runtime.replace("python", "")
        ),
        eval_protocol_install_source=".",
        user_requirements_path=None,  # Explicitly None, inline_requirements_content will be used
        inline_requirements_content=inline_requirements_content,
        service_port=8080,
    )
    if not dockerfile_content:
        print("Failed to generate Dockerfile content. Aborting.")
        return None

    image_tag = "latest"
    image_name_tag = f"{gcp_region}-docker.pkg.dev/{gcp_project_id}/{gcp_ar_repo_name}/{args.id}:{image_tag}"
    build_context_dir = os.getcwd()

    if not build_and_push_docker_image(
        image_name_tag=image_name_tag,
        dockerfile_content=dockerfile_content,
        build_context_dir=build_context_dir,
        gcp_project_id=gcp_project_id,
    ):
        print(f"Failed to build and push Docker image {image_name_tag}. Aborting.")
        return None
    print(f"Successfully built and pushed Docker image: {image_name_tag}")

    gcp_env_vars: Dict[str, str] = {}
    parsed_gcp_secrets: Dict[str, Any] = {}
    allow_unauthenticated_gcp = True

    resolved_auth_mode = "api-key"
    if gcp_config_from_yaml and gcp_config_from_yaml.default_auth_mode:
        resolved_auth_mode = gcp_config_from_yaml.default_auth_mode
    if args.gcp_auth_mode is not None:
        resolved_auth_mode = args.gcp_auth_mode
    print(f"Using GCP Auth Mode for service: {resolved_auth_mode}")

    if resolved_auth_mode == "api-key":
        print("Configuring GCP Cloud Run service for API key authentication (application layer).")
        evaluator_id = args.id
        api_key_for_service = None  # This is the key the service itself will use
        config_path = global_loaded_config_path

        if current_config.evaluator_endpoint_keys and evaluator_id in current_config.evaluator_endpoint_keys:
            api_key_for_service = current_config.evaluator_endpoint_keys[evaluator_id]
            print(f"Using existing API key for '{evaluator_id}' from configuration for the service.")
        else:
            api_key_for_service = secrets.token_hex(32)
            print(f"Generated new API key for '{evaluator_id}' for the service.")
            if not current_config.evaluator_endpoint_keys:
                current_config.evaluator_endpoint_keys = {}
            current_config.evaluator_endpoint_keys[evaluator_id] = api_key_for_service
            if config_path:
                _save_config(current_config, config_path)
            else:
                print(f"Warning: No rewardkit.yaml found to save API key for '{evaluator_id}'.")

        gcp_sanitized_eval_id = "".join(filter(lambda char: char.isalnum() or char in ["-", "_"], args.id))
        if not gcp_sanitized_eval_id:
            gcp_sanitized_eval_id = "evalprotocol-evaluator"
        secret_id_for_auth_key = f"rk-eval-{gcp_sanitized_eval_id}-authkey"
        secret_labels = {"managed-by": "eval-protocol", "evaluator-id": evaluator_id}

        api_key_secret_version_id = ensure_gcp_secret(
            project_id=gcp_project_id,
            secret_id=secret_id_for_auth_key,
            secret_value=api_key_for_service,
            labels=secret_labels,
        )
        if not api_key_secret_version_id:
            print(f"Error: Failed to store API key in GCP Secret Manager for '{evaluator_id}'. Aborting.")
            return None
        print(f"API key for service stored in GCP Secret Manager: {secret_id_for_auth_key}")
        parsed_gcp_secrets["RK_ENDPOINT_API_KEY"] = api_key_secret_version_id

        # Register this key with Fireworks secrets for the shim
        fireworks_account_id_for_secret = get_fireworks_account_id()
        if fireworks_account_id_for_secret:
            fw_eval_id_sanitized = args.id.lower()
            fw_eval_id_sanitized = "".join(filter(lambda char: char.isalnum() or char == "-", fw_eval_id_sanitized))
            fw_eval_id_sanitized = "-".join(filter(None, fw_eval_id_sanitized.split("-")))
            if not fw_eval_id_sanitized:
                fw_eval_id_sanitized = "evaluator"
            fw_eval_id_sanitized = fw_eval_id_sanitized[:40]
            fw_secret_key_name = f"rkeval-{fw_eval_id_sanitized}-shim-key"
            print(f"Registering API key on Fireworks platform as secret '{fw_secret_key_name}' for shim...")
            if create_or_update_fireworks_secret(
                account_id=fireworks_account_id_for_secret,
                key_name=fw_secret_key_name,
                secret_value=api_key_for_service,
            ):
                print(f"Successfully registered/updated secret '{fw_secret_key_name}' on Fireworks platform.")
            else:
                print(f"Warning: Failed to register/update secret '{fw_secret_key_name}' on Fireworks platform.")
        else:
            print("Warning: Fireworks Account ID not found, cannot store shim API key on Fireworks platform.")

    cloud_run_service_url = deploy_to_cloud_run(
        service_name=args.id,
        image_name_tag=image_name_tag,
        gcp_project_id=gcp_project_id,
        gcp_region=gcp_region,
        allow_unauthenticated=allow_unauthenticated_gcp,  # True if api-key mode, app handles auth
        env_vars=gcp_env_vars if gcp_env_vars else None,
        secrets_to_mount=parsed_gcp_secrets,
    )

    if not cloud_run_service_url:
        print("Failed to deploy to Cloud Run or retrieve service URL. Aborting.")
        return None

    print(f"Successfully deployed to Cloud Run. Service URL: {cloud_run_service_url}")
    return cloud_run_service_url


# Helper to save config (can be moved to config.py later)
def _save_config(config_data: RewardKitConfig, path: str):
    # Basic save, ideally config.py would provide a robust method
    try:
        with open(path, "w") as f:
            yaml.dump(config_data.model_dump(exclude_none=True), f, sort_keys=False)
        print(f"Config updated and saved to {path}")
    except Exception as e:
        print(f"Warning: Failed to save updated config to {path}: {e}")


def deploy_command(args):
    """Create and deploy an evaluator or register a remote one."""

    # Check environment variables
    if not check_environment():
        return 1

    if not args.id:  # ID is always required
        print("Error: Evaluator ID (--id) is required.")
        return 1

    # Process HuggingFace key mapping if provided
    huggingface_message_key_map = None
    if args.huggingface_key_map:
        try:
            huggingface_message_key_map = json.loads(args.huggingface_key_map)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for --huggingface-key-map")
            return 1

    # Initialize variables for URL registration path
    service_url_to_register = None
    # api_key_for_shim = None # Not currently used by create_evaluation for shim auth directly

    # PIDs for cleanup if registration fails for local-serve
    local_server_pid_to_clean = None
    # serveo_pid_to_clean = None # This was old, replaced by local_tunnel_pid_to_clean
    local_tunnel_pid_to_clean = None  # Initialize here

    if args.target == "gcp-cloud-run":
        current_config = get_config()  # Needed by the helper
        gcp_config_from_yaml = current_config.gcp_cloud_run if current_config.gcp_cloud_run else None

        cloud_run_service_url = _deploy_to_gcp_cloud_run(args, current_config, gcp_config_from_yaml)
        if not cloud_run_service_url:
            return 1  # Error already printed by helper
        service_url_to_register = cloud_run_service_url

    elif args.target == "local-serve":
        # Renamed helper and updated return values
        url, tunnel_provider, server_pid, tunnel_pid = _establish_local_server_and_tunnel(args)
        if not url:
            # _establish_local_server_and_tunnel handles cleanup of server if tunnel fails completely
            return 1  # Error already printed by helper
        service_url_to_register = url
        local_server_pid_to_clean = server_pid
        # serveo_pid_to_clean was specific, now it's generic tunnel_pid
        # Let's rename it for clarity in the cleanup logic
        local_tunnel_pid_to_clean = tunnel_pid
        print(f"Tunnel established using {tunnel_provider}.")

    elif args.remote_url:
        # This is for --target fireworks (default) but with --remote-url
        print(f"Registering remote URL: {args.remote_url} for evaluator '{args.id}'")
        if not (args.remote_url.startswith("http://") or args.remote_url.startswith("https://")):
            print(f"Error: Invalid --remote-url '{args.remote_url}'. Must start with http:// or https://")
            return 1
        if args.metrics_folders:  # This check might be redundant if --target is explicit
            print("Info: --metrics-folders are ignored when deploying with --remote-url.")
        service_url_to_register = args.remote_url
        # No specific shim auth provided by this path.

    # Common registration step for targets that produce a URL
    if service_url_to_register:
        try:
            print(f"Registering URL '{service_url_to_register}' with Fireworks AI for evaluator '{args.id}'...")
            evaluator = create_evaluation(
                evaluator_id=args.id,
                remote_url=service_url_to_register,
                display_name=args.display_name or args.id,
                description=args.description or f"Evaluator for {args.id} at {service_url_to_register}",
                force=args.force,
                huggingface_dataset=args.huggingface_dataset,
                huggingface_split=args.huggingface_split,
                huggingface_message_key_map=huggingface_message_key_map,
                huggingface_prompt_key=args.huggingface_prompt_key,
                huggingface_response_key=args.huggingface_response_key,
                # remote_auth_header_name="X-Api-Key" if api_key_for_shim else None, # No API key for shim for now
                # remote_auth_header_value=api_key_for_shim # No API key for shim for now
            )
            evaluator_name = evaluator.get("name", args.id)
            print(
                f"Successfully registered evaluator '{evaluator_name}' on Fireworks AI, pointing to '{service_url_to_register}'."
            )
            if args.target == "local-serve":
                # tunnel_provider is defined in the local-serve block
                # We need to ensure it's accessible here or pass it through.
                # For now, let's assume tunnel_provider was defined in the calling scope of this block.
                # This will require a small adjustment to how tunnel_provider is scoped.
                # Let's fetch it from args if we store it there, or pass it.
                # Simpler: just make the message generic or re-fetch from the PIDs.
                # The variable `tunnel_provider` is set in the `elif args.target == "local-serve":` block.
                # It needs to be available here.
                # For now, I'll adjust the print statement to be more generic or rely on the PIDs.
                # The `tunnel_provider` variable is indeed set in the correct scope.
                print(
                    f"Local server (PID: {local_server_pid_to_clean}) and {tunnel_provider} tunnel (PID: {local_tunnel_pid_to_clean}) are running."
                )
                print("They will be stopped automatically when this command exits (e.g., Ctrl+C).")
            return 0
        except PlatformAPIError as e:
            print(f"Error registering URL with Fireworks AI: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred during Fireworks AI registration: {str(e)}")
        finally:
            # If registration fails for local-serve, clean up the started processes
            if args.target == "local-serve" and ("evaluator" not in locals() or not locals().get("evaluator")):
                print("Registration failed or was interrupted for local-serve. Cleaning up local processes...")
                if local_tunnel_pid_to_clean:  # Use the new generic tunnel PID variable
                    stop_process(local_tunnel_pid_to_clean)
                if local_server_pid_to_clean:
                    stop_process(local_server_pid_to_clean)
        return 1

    # Fallback to original behavior: Deploying by packaging local metrics_folders (target=fireworks, no remote_url)
    # This is when args.target == "fireworks" (default) AND args.remote_url is NOT provided.
    elif args.target == "fireworks" and not args.remote_url:
        if not args.metrics_folders:
            print("Error: --metrics-folders are required for 'fireworks' target if --remote-url is not provided.")
            return 1
        for folder_spec in args.metrics_folders:
            if "=" not in folder_spec:
                print(f"Error: Metric folder format should be 'name=path', got '{folder_spec}'")
                return 1
        try:
            print(f"Packaging and deploying metrics for evaluator '{args.id}' to Fireworks AI...")
            evaluator = create_evaluation(
                evaluator_id=args.id,
                metric_folders=args.metrics_folders,
                display_name=args.display_name or args.id,
                description=args.description or f"Evaluator: {args.id}",
                force=args.force,
                huggingface_dataset=args.huggingface_dataset,
                huggingface_split=args.huggingface_split,
                huggingface_message_key_map=huggingface_message_key_map,
                huggingface_prompt_key=args.huggingface_prompt_key,
                huggingface_response_key=args.huggingface_response_key,
            )
            evaluator_name = evaluator.get("name", args.id)
            print(f"Successfully created/updated evaluator: {evaluator_name}")
            return 0
        except PlatformAPIError as e:
            print(f"Error creating/updating evaluator '{args.id}': {str(e)}")
            return 1
        except Exception as e:
            print(f"Error creating/updating evaluator '{args.id}': {str(e)}")
            return 1
