"""
SVGBench evaluation test for EvalProtocol.io using RemoteRolloutProcessor.

This test evaluates LLM ability to generate SVG code that meets specific visual requirements.
The remote server handles:
1. SVG code generation from text prompts (model calls)

The local test handles:
2. SVG to PNG rendering using Selenium
3. LLM judge evaluation of requirement fulfillment
4. Scoring based on fulfilled requirements ratio
"""

import base64
import json
import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List
import asyncio
import pytest

import litellm
from pydantic import BaseModel

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor

from utils import extract_svg_code, render_svg_to_png

logger = logging.getLogger(__name__)


class SVGBenchResponse(BaseModel):
    reasoning: str
    number_of_fulfilled_requirements: int


async def evaluate_with_llm_judge(image_path: str, requirements: List[str]) -> Dict[str, Any]:
    """
    Use LLM judge to evaluate how many requirements are fulfilled.
    Uses GPT-4.1 for vision capabilities to match project's model preferences. (note original repo uses Gemini 2.5 flashs)

    Args:
        image_path: Path to rendered PNG image
        requirements: List of requirements to evaluate

    Returns:
        Dictionary with evaluation results
    """
    # Format requirements for evaluation (exactly as in original)
    requirements_text = "\n".join([f"{i + 1}. {req}" for i, req in enumerate(requirements)])

    # Create evaluation prompt with JSON response format
    evaluate_prompt = f"""Examine the generated image. How many of the following {len(requirements)} requirements were fulfilled?

Be strict about the requirements and respond ONLY with a JSON object in this exact format:
{{"reasoning": <reasoning_text>,
"number_of_fulfilled_requirements": <count>}}

Where <count> is a number between 0 and {len(requirements)}.

Requirements:
{requirements_text}"""

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Prepare messages with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": evaluate_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }
    ]

    # Use GPT-4.1 for vision capabilities to match project's OpenAI model preference
    response = await litellm.acompletion(
        model="gpt-4.1",
        messages=messages,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "SVGBenchResponse", "schema": SVGBenchResponse.model_json_schema()},
        },
    )

    # Parse response
    response_content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]

    # Handle empty response
    if not response_content or response_content.strip() == "":
        raise ValueError("Empty response from LLM judge")

    result = json.loads(response_content)

    # Validate the result
    if "number_of_fulfilled_requirements" in result:
        return result
    else:
        raise ValueError("Missing required field in response")


@pytest.mark.skip(reason="Skipping SVG generation evaluation test")
@evaluation_test(
    input_dataset=[str(Path(__file__).parent / "svgbench_dataset.jsonl")],
    completion_params=[
        {
            "temperature": 0.8,
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
            "extra_body": {"reasoning_effort": "medium"},
        },
    ],
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="https://vercel-svg-server-ts.vercel.app",
    ),
    passed_threshold=0.5,
    max_dataset_rows=8,
    num_runs=1,
    mode="pointwise",
)
async def test_svg_generation_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    SVG generation evaluation.

    This evaluation asks: How many of the requirements were fulfilled?
    """
    assert row.input_metadata.dataset_info is not None

    # Extract dataset info
    requirements = row.input_metadata.dataset_info["requirements"]
    total_requirements = row.input_metadata.dataset_info["total_requirements"]
    original_prompt = row.input_metadata.dataset_info["original_prompt"]
    row_id = row.input_metadata.row_id

    # Check if we should save debug files
    save_debug_files = os.environ.get("SVGBENCH_SAVE_DEBUG_FILES", "false").lower() == "true"

    # Get model response
    if not row.messages or len(row.messages) < 2:
        row.evaluation_result = EvaluateResult(score=0.0, reason="No model response found", is_score_valid=False)
        return row

    model_response = row.messages[-1].content
    assert isinstance(model_response, str)

    # Extract SVG code
    try:
        svg_code = extract_svg_code(model_response)
        if not svg_code:
            raise ValueError("No valid SVG code found in response")
    except Exception as e:
        logger.error(f"Error extracting SVG code for question {row_id}: {e}")
        row.evaluation_result = EvaluateResult(score=0.0, reason=f"SVG extraction failed: {str(e)}")
        return row

    # Setup file paths
    if save_debug_files:
        model = row.input_metadata.completion_params["model"]
        safe_model_name = model.replace("/", "_").replace(":", "_")
        debug_dir = "svgbench_debug_intent_matching"
        os.makedirs(debug_dir, exist_ok=True)
        png_path = os.path.join(debug_dir, f"question_{row_id}_{safe_model_name}.png")
        svg_path = os.path.join(debug_dir, f"question_{row_id}_{safe_model_name}.svg")
        with open(svg_path, "w") as f:
            f.write(svg_code)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

    try:
        # Render SVG to PNG
        try:
            svg_render_success = await asyncio.to_thread(render_svg_to_png, svg_code, png_path)
            if not svg_render_success:
                row.evaluation_result = EvaluateResult(
                    score=0.0,
                    reason="Failed to render SVG to PNG - render_svg_to_png returned False",
                    is_score_valid=False,
                )
                return row
        except Exception as e:
            # Capture full stack trace for debugging
            full_traceback = traceback.format_exc()
            error_reason = f"Failed to render SVG to PNG - Exception occurred:\n\nError: {str(e)}\n\nFull Stack Trace:\n{full_traceback}"
            row.evaluation_result = EvaluateResult(score=0.0, reason=error_reason, is_score_valid=False)
            return row

        # Run LLM judge evaluation
        judge_result = await evaluate_with_llm_judge(png_path, requirements)

        # Calculate score
        fulfilled_count = judge_result.get("number_of_fulfilled_requirements", 0)
        fulfilled_count = max(0, min(fulfilled_count, total_requirements))  # Clamp to valid range
        score = fulfilled_count / total_requirements

        row.evaluation_result = EvaluateResult(
            score=score,
            reason=judge_result.get("reasoning", ""),
        )

        return row

    except Exception as e:
        logger.error(f"LLM judge evaluation failed for question {row_id}: {e}")
        row.evaluation_result = EvaluateResult(score=0.0, reason=f"Evaluation error: {str(e)}", is_score_valid=False)
        return row

    finally:
        # Clean up temporary PNG file (only if not saving debug files)
        if not save_debug_files:
            try:
                if os.path.exists(png_path):
                    os.unlink(png_path)
            except Exception:
                pass
