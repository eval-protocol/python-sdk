import importlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, cast

import uvicorn  # type: ignore[reportMissingImports]
from fastapi import FastAPI, HTTPException, Request  # type: ignore[reportMissingImports]
from pydantic import BaseModel, Field  # type: ignore[reportMissingImports]

from .models import EvaluateResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Model for a conversation message."""

    role: str
    content: str

    class Config:
        extra = "allow"  # Allow extra fields


class RewardRequest(BaseModel):
    """Request model for reward endpoints."""

    messages: List[Message] = Field(..., description="List of conversation messages")
    ground_truth: Optional[Union[str, List[Message]]] = Field(
        None, description="Ground truth data (string or list of messages) for context"
    )

    class Config:
        extra = "allow"  # Allow extra fields for arbitrary kwargs


class RewardServer:
    """
    Server for hosting reward functions.

    This class creates a FastAPI server that can host reward functions.

    Args:
        func_path: Path to the reward function to host (e.g., "module.path:function_name")
        host: Host to bind the server to
        port: Port to bind the server to
    """

    def __init__(
        self,
        func_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.func_path = func_path
        self.host = host
        self.port = port
        self.app = FastAPI(title="Reward Function Server")

        # Load the reward function
        self.reward_func = self._load_function()

        # Register the endpoints
        self._setup_routes()

    def _load_function(self):
        """Load the reward function from the provided path."""
        try:
            if ":" not in self.func_path:
                raise ValueError(f"Invalid func_path format: {self.func_path}, expected 'module.path:function_name'")

            module_path, func_name = self.func_path.split(":", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)

            logger.info(f"Loaded reward function {func_name} from {module_path}")
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load function from path {self.func_path}: {str(e)}")

    def _setup_routes(self):
        """Set up the API routes."""

        @self.app.get("/")
        async def root():
            """Get server info."""
            return {
                "status": "ok",
                "reward_function": self.func_path,
                "endpoints": ["/reward"],
            }

        @self.app.post("/reward")
        async def reward(request: RewardRequest):
            """
            Get reward score for messages.

            Args:
                request: RewardRequest object with messages and optional parameters

            Returns:
                EvaluateResult object with score and metrics
            """
            try:
                # Extract kwargs from the request
                kwargs = request.dict(exclude={"messages", "ground_truth"})

                # Set default for ground_truth if not provided and expected as list
                ground_truth_data = request.ground_truth
                if ground_truth_data is None:
                    # This default applies if ground_truth is expected to be a list of messages for context
                    ground_truth_data = request.messages[:-1] if request.messages else []

                # Call the reward function
                result = self.reward_func(
                    messages=request.messages,
                    ground_truth=ground_truth_data,
                    **kwargs,
                )

                # Handle different return types
                # The self.reward_func is expected to be decorated by the new @reward_function,
                # which returns a dictionary.
                if isinstance(result, dict) and "score" in result:
                    return result
                elif isinstance(result, EvaluateResult):  # Should not happen if func is from new decorator
                    logger.warning("Reward function returned EvaluateResult object directly to server; expected dict.")
                    return result.model_dump()
                elif isinstance(result, tuple) and len(result) == 2:  # Legacy tuple
                    logger.warning("Reward function returned legacy tuple format to server.")
                    score, components = result
                    return {"score": score, "metrics": components}
                else:
                    raise TypeError(f"Invalid return type from reward function after decoration: {type(result)}")

            except Exception as e:
                logger.error(f"Error processing reward request: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "ok"}

    def run(self):
        """Run the server."""
        logger.info(f"Starting reward server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


def serve(func_path: str, host: str = "0.0.0.0", port: int = 8000):
    """
    Serve a reward function as an HTTP API.

    Args:
        func_path: Path to the reward function to serve (e.g., "module.path:function_name")
        host: Host to bind the server to
        port: Port to bind the server to
    """
    server = RewardServer(func_path=func_path, host=host, port=port)
    server.run()


# ngrok-based serve_tunnel is deprecated in favor of Serveo via subprocess_manager.
# def serve_tunnel(func_path: str, port: int = 8000):
#     """
#     Serve a reward function with an ngrok tunnel.
#     DEPRECATED.
#     """
#     try:
#         import pyngrok.ngrok as ngrok  # type: ignore
#     except ImportError:
#         raise ImportError(
#             "The 'pyngrok' package is required to use serve_tunnel. "
#             "Please install it with 'pip install pyngrok'."
#         )
#
#     # Open the tunnel
#     tunnel = ngrok.connect(port)
#     public_url = tunnel.public_url
#
#     # Print the tunnel URL
#     logger.info(f"Reward function available at: {public_url}/reward")
#
#     # Start the server
#     serve(func_path=func_path, host="0.0.0.0", port=port)


def create_app(reward_func: Callable[..., EvaluateResult]) -> FastAPI:
    """
    Create a FastAPI app for the given reward function.

    This function creates a FastAPI app that can be used to serve a reward function.
    It's particularly useful for testing or when you want to manage the lifecycle
    of the app yourself.

    Args:
        reward_func: The reward function to serve

    Returns:
        A FastAPI app instance
    """
    app = FastAPI(title="Reward Function Server")

    @app.get("/")
    async def root():
        """Get server info."""
        return {"status": "ok", "endpoints": ["/reward"]}

    @app.post("/reward")
    async def reward(request_data: RewardRequest):
        """
        Get reward score for messages.

        Args:
            request_data: RewardRequest object with messages and optional parameters

        Returns:
            EvaluateResult object with score and metrics
        """
        try:
            # Convert Pydantic models to dictionaries using model_dump (Pydantic v2)
            messages = [msg.model_dump() for msg in request_data.messages]
            ground_truth_data: Optional[Union[str, List[Dict[str, Any]]]] = None

            if isinstance(request_data.ground_truth, str):
                ground_truth_data = request_data.ground_truth
            elif isinstance(request_data.ground_truth, list):
                ground_truth_data = [msg.model_dump() for msg in request_data.ground_truth]

            # Extract kwargs from any extra fields
            kwargs = {k: v for k, v in request_data.model_dump().items() if k not in ["messages", "ground_truth"]}

            # Set default for ground_truth if not provided and expected as list
            if ground_truth_data is None:
                # This default applies if ground_truth is expected to be a list of messages for context
                ground_truth_data = messages[:-1] if messages else []

            # Call the reward function
            result = reward_func(messages=messages, ground_truth=ground_truth_data, **kwargs)

            # Handle different return types
            # The reward_func is expected to be decorated by the new @reward_function,
            # which returns a dictionary.
            if isinstance(result, dict) and "score" in result:
                return result
            elif isinstance(result, EvaluateResult):  # Should not happen if func is from new decorator
                logger.warning(
                    "Reward function passed to create_app returned EvaluateResult object directly; expected dict after decoration."
                )
                return result.model_dump()
            elif isinstance(result, tuple) and len(result) == 2:  # Legacy tuple
                logger.warning("Reward function passed to create_app returned legacy tuple format.")
                score, components = cast(Tuple[float, Dict[str, Any]], result)
                return {"score": score, "metrics": components}
            else:
                raise TypeError(f"Invalid return type from reward function after decoration: {type(result)}")

        except Exception as e:
            logger.error(f"Error processing reward request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return app
