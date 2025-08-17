import importlib
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

# Assuming these models are correctly defined in eval_protocol.models
from eval_protocol.models import EvaluateResult, Message


# --- Request and Response Models ---
class EvaluationRequest(BaseModel):
    messages: List[Dict[str, Any]]  # Could also be List[Message] if we enforce that model on input
    ground_truth: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}


# --- Global variable to store the loaded reward function ---
# This is a simple approach for a single-function server.
# If multiple functions were to be served by one instance, a different mechanism would be needed.
_LOADED_REWARD_FUNCTION = None
_REWARD_FUNCTION_NAME = "N/A"

# --- API Key Authentication Dependency ---
EXPECTED_API_KEY = os.environ.get("RK_ENDPOINT_API_KEY")


async def verify_api_key(request: Request):
    if EXPECTED_API_KEY:
        # Check for X-Api-Key header first
        client_api_key = request.headers.get("X-Api-Key")
        # If not found, check for Authorization: Bearer <key>
        if not client_api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                client_api_key = auth_header.split(" ", 1)[1]

        if not client_api_key:
            raise HTTPException(status_code=401, detail="API key required but not provided.")
        if client_api_key != EXPECTED_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key.")
    return True  # Allow request if no key expected or if key is valid


# --- FastAPI App ---
app = FastAPI(
    title="Eval Protocol Generic Reward Function Server",
    description="Serves a dynamically loaded reward function.",
    version="0.1.0",  # Or use eval_protocol.__version__
)


@app.post("/evaluate", response_model=EvaluateResult, dependencies=[Depends(verify_api_key)])
async def evaluate_endpoint(request: EvaluationRequest):
    """
    Endpoint to evaluate a given set of messages using the loaded reward function.
    Requires API key if RK_ENDPOINT_API_KEY environment variable is set.
    """
    if _LOADED_REWARD_FUNCTION is None:
        raise HTTPException(status_code=500, detail="Reward function not loaded.")

    try:
        # The user's reward function is expected to match the @reward_function signature
        func_args = {
            "messages": request.messages,
            "ground_truth": request.ground_truth,
            **(request.kwargs or {}),
        }

        result = _LOADED_REWARD_FUNCTION(**func_args)

        if not isinstance(result, EvaluateResult):
            # This case should ideally not happen if functions are correctly decorated
            # and return EvaluateResult, but good to have a fallback.
            print(
                f"Warning: Reward function '{_REWARD_FUNCTION_NAME}' did not return an EvaluateResult instance. Type: {type(result)}"
            )
            # Attempt to construct an EvaluateResult if it's a dict-like object,
            # otherwise, this will raise an error or return a poorly formed response.
            # For robustness, one might want to wrap this in another try-except.
            return EvaluateResult(
                score=0.0,
                reason="Invalid return type from reward function, check server logs.",
                is_score_valid=False,
                metrics={},
            )

        return result
    except ValidationError as ve:  # Pydantic validation error from reward function's input/output
        print(f"Validation Error calling reward function '{_REWARD_FUNCTION_NAME}': {ve}")
        raise HTTPException(
            status_code=422,
            detail=f"Input/Output validation error for reward function: {ve.errors()}",
        )
    except Exception as e:
        print(f"Error during evaluation with reward function '{_REWARD_FUNCTION_NAME}': {e}")
        # Consider logging the full traceback here
        raise HTTPException(status_code=500, detail=f"Internal server error during evaluation: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    if _LOADED_REWARD_FUNCTION:
        return {"status": "ok", "reward_function": _REWARD_FUNCTION_NAME}
    else:
        return {"status": "error", "reason": "Reward function not loaded"}


def load_reward_function(import_string: str):
    """
    Loads a reward function from an import string (e.g., 'my_module.my_function').
    """
    global _LOADED_REWARD_FUNCTION, _REWARD_FUNCTION_NAME
    try:
        module_path, function_name = import_string.rsplit(".", 1)
        module = importlib.import_module(module_path)
        _LOADED_REWARD_FUNCTION = getattr(module, function_name)
        _REWARD_FUNCTION_NAME = import_string
        print(f"Successfully loaded reward function: {_REWARD_FUNCTION_NAME}")
    except Exception as e:
        print(f"Error loading reward function from '{import_string}': {e}")
        _LOADED_REWARD_FUNCTION = None
        _REWARD_FUNCTION_NAME = "Error loading"
        raise  # Re-raise to make it fatal if loading fails on startup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Generic Reward Function Server.")
    parser.add_argument(
        "import_string",
        type=str,
        help="Import string for the reward function (e.g., 'my_package.my_module.my_reward_function')",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to.")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,  # Standard port for Cloud Run, etc.
        help="Port to bind the server to.",
    )
    # Add --reload for uvicorn if needed for development
    # parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")

    args = parser.parse_args()

    try:
        load_reward_function(args.import_string)
    except Exception:
        print("Failed to load reward function. Exiting.")
        exit(1)

    if not _LOADED_REWARD_FUNCTION:
        print(f"Reward function {_REWARD_FUNCTION_NAME} could not be loaded. Server will not start correctly.")
        # Depending on desired behavior, could exit here or let it run and fail on /evaluate
        exit(1)

    print(f"Starting server for reward function: {args.import_string} on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)  # reload=args.reload for dev
