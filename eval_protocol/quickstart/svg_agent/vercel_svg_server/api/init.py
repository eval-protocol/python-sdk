"""
Vercel serverless function for SVGBench remote evaluation.

This function handles the model call part of the evaluation pipeline.
The SVG evaluation logic remains in the test client.
"""

import json
import os
import logging
import sys
import asyncio
from flask import Flask, request, jsonify
from openai import OpenAI
import openai
from dotenv import load_dotenv

from eval_protocol import Status, InitRequest, FireworksTracingHttpHandler, RolloutIdFilter

load_dotenv()

# Configure logging so INFO and below go to stdout, WARNING+ to stderr.
# This avoids Vercel marking INFO logs as [error] (stderr).
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)


class _InfoOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= logging.INFO


formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(_InfoOnly())
stdout_handler.setFormatter(formatter)
root_logger.addHandler(stdout_handler)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.setFormatter(formatter)
root_logger.addHandler(stderr_handler)

# Attach Fireworks tracing handler to root logger (non-stream HTTP sink)
root_logger.addHandler(FireworksTracingHttpHandler())

# Create Flask app
app = Flask(__name__)


async def execute_rollout_background(req: InitRequest, api_key: str):
    """Execute the OpenAI completion in background and log results"""
    # Attach rollout_id filter to logger
    logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    model = req.completion_params.get("model")
    # Uncomment if you need to strip fireworks_ai/ prefix
    # if model and isinstance(model, str) and model.startswith("fireworks_ai/"):
    #     model = model[len("fireworks_ai/"):]

    # Prepare completion arguments
    completion_kwargs = {
        "messages": req.messages,
        # "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "model": model,
        "temperature": req.completion_params.get("temperature"),
        "max_tokens": req.completion_params.get("max_tokens"),
    }

    # Add tools if present
    if req.tools:
        completion_kwargs["tools"] = req.tools

    logger.info(
        f"DEBUG: {req.model_base_url}, COMPLETION_KWARGS: {completion_kwargs}, API_KEY: {api_key}, MODEL: {model}"
    )

    # Create AsyncOpenAI client
    # client = AsyncOpenAI(base_url=req.model_base_url, api_key=api_key)
    client = OpenAI(base_url=req.model_base_url, api_key=api_key)

    logger.info(f"Sending completion request to model {model}")

    # Make the async model call with timeout
    import time

    logger.info(f"timing start: {time.time()}")

    try:
        completion = client.chat.completions.create(**completion_kwargs)
    except (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
    ) as e:
        # These errors should be logged and will be retried by RemoteRolloutProcessor
        logger.error(
            f"Rollout {req.metadata.rollout_id} failed: {e}",
            extra={"status": Status.rollout_permission_denied_error(str(e))},
        )
        return
    except openai.NotFoundError as e:
        logger.error(
            f"Rollout {req.metadata.rollout_id} failed: {e}", extra={"status": Status.rollout_not_found_error(str(e))}
        )
        return
    except openai.RateLimitError as e:
        logger.error(
            f"Rollout {req.metadata.rollout_id} failed: {e}",
            extra={"status": Status.rollout_resource_exhausted_error(str(e))},
        )
        return
    except Exception as e:
        # Non-OpenAI errors (shouldn't normally happen but catch anyway)
        logger.error(
            f"Rollout {req.metadata.rollout_id} failed with unexpected error: {e}",
            extra={"status": Status.rollout_internal_error(str(e))},
        )
        return

    logger.info(f"Completed response: {completion}")
    logger.info(f"timing end: {time.time()}")
    # Log successful completion - THIS IS WHAT RemoteRolloutProcessor POLLS FOR
    logger.info(f"Rollout {req.metadata.rollout_id} completed", extra={"status": Status.rollout_finished()})


@app.route("/init", methods=["POST"])
async def init():
    try:
        # Parse as InitRequest
        req = InitRequest(**request.get_json())

        # Create logger for immediate validation logging
        logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
        logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

        # Validate required fields
        if not req.messages:
            error_msg = "messages is required"
            logger.error(error_msg, extra={"status": Status.rollout_internal_error(error_msg)})
            return jsonify({"error": error_msg}), 400

        # Get API key (prefer request api_key, fallback to environment)
        if req.api_key:
            logger.info("Using API key from request")
            api_key = req.api_key
        elif os.environ.get("FIREWORKS_API_KEY"):
            logger.info("Using API key from environment")
            api_key = os.environ.get("FIREWORKS_API_KEY")
        else:
            error_msg = "API key not provided in request or environment variable"
            logger.error(error_msg, extra={"status": Status.rollout_internal_error(error_msg)})
            return jsonify({"error": error_msg}), 401

        # 🔥 FIRE: Return immediately with acceptance (within 30s requirement)
        response_data = {
            "status": "accepted",
            "rollout_id": req.metadata.rollout_id,
            "message": "Rollout processing started",
        }

        # Fire and forget: Execute rollout asynchronously
        asyncio.create_task(execute_rollout_background(req, api_key or ""))

        return jsonify(response_data), 200

    except Exception as e:
        # For request parsing errors, return error immediately (don't retry)
        return jsonify({"error": f"Request parsing error: {str(e)}"}), 400


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "ok",
            "message": "SVGBench Vercel Serverless Function",
            "endpoints": {"POST /": "Process SVGBench evaluation requests"},
        }
    )


@app.route("/", methods=["OPTIONS"])
def options_handler():
    """Handle CORS preflight requests"""
    response = jsonify({})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response
