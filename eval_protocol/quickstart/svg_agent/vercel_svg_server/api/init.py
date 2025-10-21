"""
Vercel serverless function for SVGBench remote evaluation.

This function handles the model call part of the evaluation pipeline.
The SVG evaluation logic remains in the test client.
"""

import json
import os
import logging
from http.server import BaseHTTPRequestHandler
from openai import OpenAI
from dotenv import load_dotenv

from eval_protocol import Status, InitRequest, FireworksTracingHttpHandler, RolloutIdFilter

load_dotenv()

# Attach Fireworks tracing handler to root logger
fireworks_handler = FireworksTracingHttpHandler()
logging.getLogger().addHandler(fireworks_handler)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Read and parse request body
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length).decode("utf-8")
            request_data = json.loads(request_body)

            # Parse as InitRequest
            req = InitRequest(**request_data)

            # Attach rollout_id filter to logger
            logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
            logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

            # Validate required fields
            if not req.messages:
                error_msg = "messages is required"
                logger.error(error_msg, extra={"status": Status.rollout_error(error_msg)})
                self._send_error(400, error_msg)
                return

            # Prepare completion arguments
            completion_kwargs = {
                "messages": req.messages,
                **req.completion_params,
            }

            # Add tools if present
            if req.tools:
                completion_kwargs["tools"] = req.tools

            # Add completion parameters if they exist
            # if hasattr(req, 'completion_params') and req.completion_params:
            #     # Filter out any model override
            #     params = {k: v for k, v in req.completion_params.items() if k != 'model'}
            #     completion_kwargs.update(params)

            # Get API key (prefer request api_key, fallback to environment)
            api_key = req.api_key or os.environ.get("FIREWORKS_API_KEY")
            if not api_key:
                error_msg = "API key not provided in request or FIREWORKS_API_KEY environment variable"
                logger.error(error_msg, extra={"status": Status.rollout_error(error_msg)})
                self._send_error(500, error_msg)
                return

            # Create OpenAI client
            client = OpenAI(base_url=req.model_base_url, api_key=api_key)

            logger.info(f"Sending completion request to model {req.completion_params.get('model')}")

            # Make the model call
            completion = client.chat.completions.create(**completion_kwargs)

            logger.info(f"Completed response: {completion}")

            # Log completion status
            logger.info(f"Rollout {req.metadata.rollout_id} completed", extra={"status": Status.rollout_finished()})

            # Return the completion response
            response_data = {
                "status": "completed",
                "rollout_id": req.metadata.rollout_id,
                "choices": [
                    {
                        "message": {
                            "role": completion.choices[0].message.role,
                            "content": completion.choices[0].message.content,
                        }
                    }
                ],
            }

            self._send_json_response(200, response_data)

        except Exception as e:
            # Log error if we have the request context
            if "req" in locals() and "logger" in locals():
                logger.error(f"❌ Error in rollout {req.metadata.rollout_id}: {e}")
                logger.error(str(e), extra={"status": Status.rollout_error(str(e))})

            self._send_error(500, str(e))

    def do_GET(self):
        """Health check endpoint"""
        self._send_json_response(
            200,
            {
                "status": "ok",
                "message": "SVGBench Vercel Serverless Function",
                "endpoints": {"POST /": "Process SVGBench evaluation requests"},
            },
        )

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json_response(self, status_code: int, data: dict):
        """Send a JSON response"""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error(self, status_code: int, message: str):
        """Send an error response"""
        self._send_json_response(status_code, {"error": message})
