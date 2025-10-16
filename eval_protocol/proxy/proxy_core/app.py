"""
Metadata Extraction Gateway
A FastAPI service that sits in front of LiteLLM and extracts metadata from URL paths.
"""

from fastapi import FastAPI, Depends, Request, Query
from typing import Optional, List
import os
import redis
import logging
import yaml
from pathlib import Path
import sys
from contextlib import asynccontextmanager

from .models import ProxyConfig, LangfuseTracesResponse, TracesParams, ChatParams, ChatRequestHook, TracesRequestHook
from .auth import AuthProvider, NoAuthProvider
from .litellm import handle_chat_completion, proxy_to_litellm
from .langfuse import fetch_langfuse_traces, pointwise_fetch_langfuse_trace

# Configure logging before any other imports (so all modules inherit this config)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger(__name__)


def build_proxy_config(
    preprocess_chat_request: Optional[ChatRequestHook] = None,
    preprocess_traces_request: Optional[TracesRequestHook] = None,
) -> ProxyConfig:
    """Load environment and secrets, and build ProxyConfig"""
    # Env
    litellm_url = os.getenv("LITELLM_URL")
    if not litellm_url:
        raise ValueError("LITELLM_URL environment variable must be set")
    request_timeout = float(os.getenv("REQUEST_TIMEOUT", "300.0"))
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Secrets - use SECRETS_PATH env var if set, otherwise default to proxy/secrets.yaml
    secrets_path_str = os.getenv("SECRETS_PATH")
    if secrets_path_str:
        secrets_path = Path(secrets_path_str)
    else:
        secrets_path = Path(__file__).parent / "secrets.yaml"
    if not secrets_path.exists():
        raise ValueError(
            "Secrets file not found! Please create it from secrets.yaml.example:\n"
            "  cp eval_protocol/proxy/proxy_core/secrets.yaml.example eval_protocol/proxy/proxy_core/secrets.yaml\n"
            "Then add your Langfuse API keys to the secrets file"
        )
    try:
        with open(secrets_path, "r") as f:
            secrets_config = yaml.safe_load(f)
        langfuse_keys = secrets_config["langfuse_keys"]
        default_project_id = secrets_config["default_project_id"]
        logger.info(f"Loaded {len(langfuse_keys)} Langfuse project(s) from {secrets_path.name}")
    except KeyError as e:
        raise ValueError(f"Missing required key in secrets file: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid format in secrets file {secrets_path.name}: {e}")

    return ProxyConfig(
        litellm_url=litellm_url,
        request_timeout=request_timeout,
        langfuse_host=langfuse_host,
        langfuse_keys=langfuse_keys,
        default_project_id=default_project_id,
        preprocess_chat_request=preprocess_chat_request,
        preprocess_traces_request=preprocess_traces_request,
    )


def init_redis() -> redis.Redis:
    """Initialize and return a Redis client from environment variables."""
    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        raise ValueError("REDIS_HOST environment variable must be set")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD")

    try:
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password if redis_password else None,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Redis at {redis_host}:{redis_port}: {e}")


def create_app(
    auth_provider: AuthProvider = NoAuthProvider(),
    preprocess_chat_request: Optional[ChatRequestHook] = None,
    preprocess_traces_request: Optional[TracesRequestHook] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build runtime on startup
        app.state.config = build_proxy_config(preprocess_chat_request, preprocess_traces_request)
        app.state.redis = init_redis()

        try:
            yield
        finally:
            try:
                app.state.redis.close()
            except Exception:
                pass

    app = FastAPI(title="LiteLLM Metadata Proxy", lifespan=lifespan)

    def get_config(request: Request) -> ProxyConfig:
        return request.app.state.config

    def get_redis(request: Request) -> redis.Redis:
        return request.app.state.redis

    def get_traces_params(
        tags: Optional[List[str]] = Query(default=None),
        project_id: Optional[str] = None,
        limit: int = 100,
        sample_size: Optional[int] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        environment: Optional[str] = None,
        version: Optional[str] = None,
        release: Optional[str] = None,
        fields: Optional[str] = None,
        hours_back: Optional[int] = None,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        sleep_between_gets: float = 2.5,
        max_retries: int = 3,
    ) -> TracesParams:
        return TracesParams(
            tags=tags,
            project_id=project_id,
            limit=limit,
            sample_size=sample_size,
            user_id=user_id,
            session_id=session_id,
            name=name,
            environment=environment,
            version=version,
            release=release,
            fields=fields,
            hours_back=hours_back,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            sleep_between_gets=sleep_between_gets,
            max_retries=max_retries,
        )

    async def require_auth(request: Request) -> None:
        account_info = auth_provider.validate_and_return_account_info(request)
        request.state.account_id = account_info.account_id if account_info else None
        return None

    # =====================
    # Chat completion routes
    # =====================
    @app.post(
        "/project_id/{project_id}/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/chat/completions"
    )
    @app.post(
        "/v1/project_id/{project_id}/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/chat/completions"
    )
    @app.post(
        "/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/chat/completions"
    )
    @app.post(
        "/v1/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/chat/completions"
    )
    @app.post(
        "/project_id/{project_id}/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/encoded_base_url/{encoded_base_url}/chat/completions"
    )
    @app.post(
        "/v1/project_id/{project_id}/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/encoded_base_url/{encoded_base_url}/chat/completions"
    )
    @app.post(
        "/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/encoded_base_url/{encoded_base_url}/chat/completions"
    )
    @app.post(
        "/v1/rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/encoded_base_url/{encoded_base_url}/chat/completions"
    )
    async def chat_completion_with_full_metadata(
        rollout_id: str,
        invocation_id: str,
        experiment_id: str,
        run_id: str,
        row_id: str,
        request: Request,
        project_id: Optional[str] = None,
        encoded_base_url: Optional[str] = None,
        config: ProxyConfig = Depends(get_config),
        redis_client: redis.Redis = Depends(get_redis),
        _: None = Depends(require_auth),
    ):
        params = ChatParams(
            project_id=project_id,
            rollout_id=rollout_id,
            invocation_id=invocation_id,
            experiment_id=experiment_id,
            run_id=run_id,
            row_id=row_id,
            encoded_base_url=encoded_base_url,
        )
        return await handle_chat_completion(
            config=config,
            redis_client=redis_client,
            request=request,
            params=params,
        )

    @app.post("/project_id/{project_id}/chat/completions")
    @app.post("/v1/project_id/{project_id}/chat/completions")
    async def chat_completion_with_project_only(
        project_id: str,
        request: Request,
        config: ProxyConfig = Depends(get_config),
        redis_client: redis.Redis = Depends(get_redis),
        _: None = Depends(require_auth),
    ):
        params = ChatParams(project_id=project_id)
        return await handle_chat_completion(
            config=config,
            redis_client=redis_client,
            request=request,
            params=params,
        )

    # ===============
    # Traces routes
    # ===============
    @app.get("/traces", response_model=LangfuseTracesResponse)
    @app.get("/v1/traces", response_model=LangfuseTracesResponse)
    @app.get("/project_id/{project_id}/traces", response_model=LangfuseTracesResponse)
    @app.get("/v1/project_id/{project_id}/traces", response_model=LangfuseTracesResponse)
    async def get_langfuse_traces(
        request: Request,
        params: TracesParams = Depends(get_traces_params),
        project_id: Optional[str] = None,
        config: ProxyConfig = Depends(get_config),
        redis_client: redis.Redis = Depends(get_redis),
        _: None = Depends(require_auth),
    ) -> LangfuseTracesResponse:
        if project_id is not None:
            params.project_id = project_id
        return await fetch_langfuse_traces(
            config=config,
            redis_client=redis_client,
            request=request,
            params=params,
        )

    @app.get("/traces/pointwise", response_model=LangfuseTracesResponse)
    @app.get("/v1/traces/pointwise", response_model=LangfuseTracesResponse)
    @app.get("/project_id/{project_id}/traces/pointwise", response_model=LangfuseTracesResponse)
    @app.get("/v1/project_id/{project_id}/traces/pointwise", response_model=LangfuseTracesResponse)
    async def pointwise_get_langfuse_trace(
        request: Request,
        params: TracesParams = Depends(get_traces_params),
        project_id: Optional[str] = None,
        config: ProxyConfig = Depends(get_config),
        redis_client: redis.Redis = Depends(get_redis),
        _: None = Depends(require_auth),
    ) -> LangfuseTracesResponse:
        if project_id is not None:
            params.project_id = project_id
        return await pointwise_fetch_langfuse_trace(
            config=config,
            redis_client=redis_client,
            request=request,
            params=params,
        )

    # Health
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "metadata-proxy"}

    # Catch-all
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all_proxy(
        path: str,
        request: Request,
        config: ProxyConfig = Depends(get_config),
    ):
        return await proxy_to_litellm(config, path, request)

    return app
