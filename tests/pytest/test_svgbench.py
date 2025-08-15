"""
SVGBench evaluation test for EvalProtocol.io.

This test evaluates LLM ability to generate SVG code that meets specific visual requirements.
The evaluation process includes:
1. SVG code generation from text prompts
2. SVG to PNG rendering using Selenium
3. LLM judge evaluation of requirement fulfillment
4. Scoring based on fulfilled requirements ratio
"""

import base64
import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor

logger = logging.getLogger(__name__)


class SVGBenchResponse(BaseModel):
    reasoning: str
    number_of_fulfilled_requirements: int


def svgbench_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert SVGBench dataset entries to EvaluationRow objects.

    Args:
        data: List of dictionaries containing prompt and requirements

    Returns:
        List of EvaluationRow objects
    """
    rows = []

    for i, row in enumerate(data):
        # Format requirements as numbered list
        requirements = "\n".join([f"{i+1}. {req}" for i, req in enumerate(row["requirements"])])

        # Create the generation prompt following SVGBench format
        prompt = f"""{row['prompt']} Wrap the SVG code in an SVG code block following the example below.

Example:
```svg
<svg viewBox="0 0 100 100" width="100" height="100">
    <circle cx="50" cy="50" r="40" fill="red" />
</svg>
```

Requirements:
{requirements}"""

        eval_row = EvaluationRow(
            messages=[Message(role="user", content=prompt)],
            input_metadata=InputMetadata(
                row_id=f"row_{i}",
                dataset_info={
                    "original_prompt": row["prompt"],
                    "requirements": row["requirements"],
                    "total_requirements": len(row["requirements"]),
                    "formatted_prompt": prompt,
                },
            ),
        )

        rows.append(eval_row)

    return rows


def extract_svg_code(text: str) -> Optional[str]:
    """
    Extract SVG code from model response using SVGBench's extraction logic.

    Args:
        text: Raw model response text

    Returns:
        Extracted SVG code or None if not found
    """
    # First try: Look for ```svg code blocks
    if "```svg" in text:
        svg_parts = text.split("```svg")
        if len(svg_parts) > 1:
            svg_code = svg_parts[1].split("```")[0].strip()
            return svg_code

    # Second try: Look for <svg>...</svg> tags
    if "<svg" in text and "</svg>" in text:
        start = text.find("<svg")
        end = text.find("</svg>") + 6
        svg_code = text[start:end].strip()
        return svg_code

    return None


def render_svg_to_png(svg_code: str, output_path: str) -> bool:
    """
    Render SVG code to PNG using Selenium WebDriver.

    Args:
        svg_code: Valid SVG code
        output_path: Path where PNG should be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if selenium and webdriver are available
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.support.ui import WebDriverWait
        except ImportError:
            logger.error("Selenium not available. Install with: pip install selenium")
            return False

        # Parse SVG dimensions
        width, height = 800, 600  # Default dimensions

        # Try to extract dimensions from SVG
        width_match = re.search(r'width="(\d+)"', svg_code)
        height_match = re.search(r'height="(\d+)"', svg_code)
        viewbox_match = re.search(r'viewBox="[^"]*?(\d+)\s+(\d+)"', svg_code)

        if width_match and height_match:
            width, height = int(width_match.group(1)), int(height_match.group(1))
        elif viewbox_match:
            width, height = int(viewbox_match.group(1)), int(viewbox_match.group(2))

        # Create HTML wrapper
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ margin: 0; padding: 20px; background: white; }}
                svg {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            {svg_code}
        </body>
        </html>
        """

        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--window-size={width+40},{height+40}")

        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            html_path = f.name

        try:
            # Initialize WebDriver
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(f"file://{html_path}")

            # Wait for SVG to load
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "svg")))

            # Take screenshot
            driver.save_screenshot(output_path)
            driver.quit()

            return True

        finally:
            # Clean up temporary file
            os.unlink(html_path)

    except Exception as e:
        logger.error(f"SVG rendering failed: {e}")
        return False


def evaluate_with_llm_judge(image_path: str, requirements: List[str]) -> Dict[str, Any]:
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
    requirements_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requirements)])

    # Create evaluation prompt with JSON response format
    evaluate_prompt = f"""Examine the generated image. How many of the following {len(requirements)} requirements were fulfilled?

Be strict about the requirements and respond ONLY with a JSON object in this exact format:
{{"number_of_fulfilled_requirements": <count>,
"reasoning": <reasoning_text>}}

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
    response = litellm.completion(
        model="gpt-4.1",
        messages=messages,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "SVGBenchResponse", "schema": SVGBenchResponse.model_json_schema()},
        },
    )

    # Parse response
    response_content = response.choices[0].message.content

    # Handle empty response
    if not response_content or response_content.strip() == "":
        raise ValueError("Empty response from LLM judge")

    result = json.loads(response_content)

    # Validate the result
    if "number_of_fulfilled_requirements" in result:
        return result
    else:
        raise ValueError("Missing required field in response")


@evaluation_test(
    input_dataset=["tests/pytest/data/svgbench_dataset.jsonl"],
    dataset_adapter=svgbench_to_evaluation_row,
    completion_params=[
        {"temperature": 0.0, "model": "gpt-4.1"},
        {
            "temperature": 0.8,
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
            "extra_body": {"reasoning_effort": "high"},
        },
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=0.5,  # 50% average score to pass
    num_runs=1,
    mode="pointwise",
    max_concurrent_rollouts=50,
)
def test_svg_generation_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Test SVG generation and evaluation using SVGBench methodology.

    This test:
    1. Extracts SVG code from the model's response
    2. Renders SVG to PNG using Selenium
    3. Uses LLM judge to evaluate requirement fulfillment
    4. Calculates score based on fulfilled requirements ratio

    Args:
        row: EvaluationRow with model's SVG generation response

    Returns:
        EvaluationRow with evaluation results
    """
    # Extract dataset info
    requirements = row.input_metadata.dataset_info["requirements"]
    total_requirements = row.input_metadata.dataset_info["total_requirements"]
    original_prompt = row.input_metadata.dataset_info["original_prompt"]
    row_id = row.input_metadata.row_id

    # Check if we should save debug files
    save_debug_files = os.environ.get("SVGBENCH_SAVE_DEBUG_FILES", "false").lower() == "true"

    # Get model response
    if not row.messages or len(row.messages) < 2:
        row.evaluation_result = EvaluateResult(score=0.0, reason="No model response found")
        return row

    model_response = row.messages[-1].content

    # Extract SVG code with better error reporting (matching original)
    try:
        svg_code = extract_svg_code(model_response)
        if not svg_code:
            raise ValueError("No valid SVG code found in response")
    except Exception as e:
        logger.error(f"Error extracting SVG code for question {row_id}: {e}")
        if save_debug_files:
            logger.error(f"Full response: {model_response}")

        row.evaluation_result = EvaluateResult(score=0.0, reason=f"SVG extraction failed: {str(e)}")
        return row

    # Setup file paths
    if save_debug_files:
        # Create debug directory
        model = row.input_metadata.completion_params["model"]
        # Sanitize model name for filesystem (replace slashes with underscores)
        safe_model_name = model.replace("/", "_").replace(":", "_")
        debug_dir = "svgbench_debug"
        os.makedirs(debug_dir, exist_ok=True)
        png_path = os.path.join(debug_dir, f"question_{row_id}_{safe_model_name}.png")
        svg_path = os.path.join(debug_dir, f"question_{row_id}_{safe_model_name}.svg")
        # Save SVG file for debugging
        with open(svg_path, "w") as f:
            f.write(svg_code)
    else:
        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

    try:
        # Render SVG to PNG
        if not render_svg_to_png(svg_code, png_path):
            row.evaluation_result = EvaluateResult(score=0.0, reason="Failed to render SVG to PNG")
            return row

        # Evaluate with LLM judge
        judge_result = evaluate_with_llm_judge(png_path, requirements)

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
        logger.error(f"Evaluation failed for question {row_id}: {e}")
        row.evaluation_result = EvaluateResult(score=0.0, reason=f"Evaluation error: {str(e)}")
        return row

    finally:
        # Clean up temporary PNG file (only if not saving debug files)
        if not save_debug_files:
            try:
                if os.path.exists(png_path):
                    os.unlink(png_path)
            except Exception:
                pass
