"""
CLI command for deploying MCP servers to Google Cloud Run.
"""

import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from eval_protocol.config import (
    GCPCloudRunConfig,
    RewardKitConfig,
    _config_file_path as global_loaded_config_path,
    get_config,
)
from eval_protocol.gcp_tools import (
    build_and_push_docker_image,
    deploy_to_cloud_run,
    ensure_artifact_registry_repo_exists,
)

from .common import check_environment


def _generate_mcp_dockerfile_content(
    mcp_server_module: str,
    python_version: str = "3.11",
    service_port: int = 8000,
    additional_requirements: Optional[str] = None,
) -> str:
    """
    Generate Dockerfile content for MCP server deployment.

    Args:
        mcp_server_module: The Python module containing the MCP server (e.g., 'frozen_lake_mcp_server')
        python_version: Python version to use in the container
        service_port: Port the MCP server will listen on
        additional_requirements: Additional pip requirements

    Returns:
        Dockerfile content as string
    """

    # Base requirements for MCP servers - matching setup.py dependencies
    base_requirements = [
        "fastmcp>=0.1.0",
        # Core Eval Protocol dependencies from setup.py
        "requests>=2.25.0",
        "pydantic>=2.0.0",
        "dataclasses-json>=0.5.7",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "openai==1.78.1",
        "aiosqlite",
        "aiohttp",
        "mcp>=1.9.2",
        "PyYAML>=5.0",
        "datasets==3.6.0",
        "fsspec==2025.3.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "gymnasium>=0.29.0",
        "httpx>=0.24.0",
        "fireworks-ai>=0.17.19",
    ]

    if additional_requirements:
        requirements = base_requirements + [req.strip() for req in additional_requirements.split("\n") if req.strip()]
    else:
        requirements = base_requirements

    # Generate pip install lines with proper escaping
    pip_install_lines = []
    for req in requirements[:-1]:
        pip_install_lines.append(f"    {req} \\")
    pip_install_lines.append(f"    {requirements[-1]}")
    pip_install_section = "\n".join(pip_install_lines)

    dockerfile_content = f"""# Multi-stage build for MCP server deployment
FROM python:{python_version}-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install MCP server requirements
RUN pip install --no-cache-dir \\
{pip_install_section}

# Production stage
FROM python:{python_version}-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python{python_version}/site-packages /usr/local/lib/python{python_version}/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the MCP server code
COPY . .

# Set environment variables for Cloud Run
# FastMCP uses HOST and PORT environment variables for streamable-http transport
ENV HOST=0.0.0.0
ENV PORT={service_port}
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE {service_port}

# Run the MCP server with proper host and port for Cloud Run
CMD ["python", "-m", "{mcp_server_module}", "--host", "0.0.0.0", "--port", "{service_port}"]
"""

    return dockerfile_content


def _deploy_mcp_to_gcp_cloud_run(args, current_config, gcp_config_from_yaml):
    """Deploy MCP server to GCP Cloud Run."""
    print(f"Starting MCP server deployment to GCP Cloud Run for '{args.id}'...")

    # Validate required arguments - either dockerfile or mcp-server-module must be provided
    if not args.dockerfile and not args.mcp_server_module:
        print("Error: Either --dockerfile or --mcp-server-module is required for MCP server deployment.")
        return None

    # Resolve GCP configuration
    gcp_project_id = args.gcp_project
    if not gcp_project_id and gcp_config_from_yaml:
        gcp_project_id = gcp_config_from_yaml.project_id
    if not gcp_project_id:
        print("Error: GCP Project ID must be provided via --gcp-project or rewardkit.yaml.")
        return None

    gcp_region = args.gcp_region
    if not gcp_region and gcp_config_from_yaml:
        gcp_region = gcp_config_from_yaml.region
    if not gcp_region:
        print("Error: GCP Region must be provided via --gcp-region or rewardkit.yaml.")
        return None

    gcp_ar_repo_name = args.gcp_ar_repo
    if not gcp_ar_repo_name and gcp_config_from_yaml:
        gcp_ar_repo_name = gcp_config_from_yaml.artifact_registry_repository
    if not gcp_ar_repo_name:
        gcp_ar_repo_name = "eval-protocol-mcp-servers"

    print(f"Using GCP Project: {gcp_project_id}, Region: {gcp_region}, AR Repo: {gcp_ar_repo_name}")

    # Ensure Artifact Registry repository exists
    if not ensure_artifact_registry_repo_exists(
        project_id=gcp_project_id, region=gcp_region, repo_name=gcp_ar_repo_name
    ):
        print(f"Failed to ensure Artifact Registry repository '{gcp_ar_repo_name}' exists. Aborting.")
        return None

    # Determine Dockerfile content - use provided file or generate
    dockerfile_content = None
    if hasattr(args, "dockerfile") and args.dockerfile:
        # Use provided Dockerfile
        dockerfile_path = Path(args.dockerfile)
        if not dockerfile_path.exists():
            print(f"Error: Dockerfile not found at {dockerfile_path}")
            return None
        print(f"Using provided Dockerfile: {dockerfile_path}")
        try:
            with open(dockerfile_path, "r") as f:
                dockerfile_content = f.read()
        except Exception as e:
            print(f"Error reading Dockerfile at {dockerfile_path}: {e}")
            return None
    else:
        # Generate Dockerfile content (legacy approach)
        print("Generating Dockerfile content from mcp-server-module...")
        dockerfile_content = _generate_mcp_dockerfile_content(
            mcp_server_module=args.mcp_server_module,
            python_version=getattr(args, "python_version", "3.11"),
            service_port=getattr(args, "port", 8000),
            additional_requirements=getattr(args, "requirements", None),
        )

    if not dockerfile_content:
        print("Failed to obtain Dockerfile content. Aborting.")
        return None

    # Build and push Docker image
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

    # Deploy to Cloud Run
    service_port = getattr(args, "port", 8000)
    env_vars = {}

    # Add any custom environment variables
    if hasattr(args, "env_vars") and args.env_vars:
        for env_pair in args.env_vars:
            if "=" in env_pair:
                key, value = env_pair.split("=", 1)
                env_vars[key] = value

    cloud_run_service_url = deploy_to_cloud_run(
        service_name=args.id,
        image_name_tag=image_name_tag,
        gcp_project_id=gcp_project_id,
        gcp_region=gcp_region,
        allow_unauthenticated=True,  # MCP servers typically need to be publicly accessible
        env_vars=env_vars if env_vars else None,
        service_port=service_port,
    )

    if not cloud_run_service_url:
        print("Failed to deploy to Cloud Run or retrieve service URL. Aborting.")
        return None

    print("🚀 Successfully deployed MCP server to Cloud Run!")
    print(f"📍 Service URL: {cloud_run_service_url}")
    print(f"🔗 MCP Connection URL: {cloud_run_service_url}")
    print(f"📋 Service Name: {args.id}")
    deployment_method = (
        "local Dockerfile" if (hasattr(args, "dockerfile") and args.dockerfile) else "auto-generated Dockerfile"
    )
    print(f"🐳 Deployment Method: {deployment_method}")
    print()
    print("🎯 Next steps:")
    print(f"   1. Test your MCP server: curl {cloud_run_service_url}/health")
    print(f"   2. Connect MCP clients to: {cloud_run_service_url}")
    print(
        f"   3. Monitor logs: gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name={args.id}' --project {gcp_project_id}"
    )

    return cloud_run_service_url


def deploy_mcp_command(args):
    """Main entry point for MCP server deployment command."""

    # Check environment (similar to existing deploy command)
    if not check_environment():
        print("Environment check failed. Please resolve the issues above before deploying.")
        return False

    try:
        # Load configuration
        current_config = get_config()
        gcp_config_from_yaml: Optional[GCPCloudRunConfig] = None
        if current_config and current_config.gcp_cloud_run:
            gcp_config_from_yaml = current_config.gcp_cloud_run

        # Deploy to GCP Cloud Run
        service_url = _deploy_mcp_to_gcp_cloud_run(args, current_config, gcp_config_from_yaml)

        if service_url:
            print(f"✅ MCP server '{args.id}' successfully deployed!")
            return True
        else:
            print(f"❌ Failed to deploy MCP server '{args.id}'")
            return False

    except Exception as e:
        print(f"Error during MCP server deployment: {e}")
        import traceback

        traceback.print_exc()
        return False
