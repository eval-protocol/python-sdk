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
import time
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message, MetricResult
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


class HumanPreferenceResponse(BaseModel):
    """Response structure for human preference evaluation with detailed rubrics."""

    intent_reasoning: str
    intent_matching_score: float  # 0-1: Does the content match the intended purpose?

    content_reasoning: str
    content_recognizability_score: float  # 0-1: Are key elements actually recognizable?

    spatial_reasoning: str
    spatial_design_score: float  # 0-1: Quality of layout, hierarchy, professional appearance

    ux_reasoning: str
    user_experience_score: float  # 0-1: Would humans find this usable/appropriate?

    coherence_reasoning: str
    visual_coherence_score: float  # 0-1: Do all elements work together harmoniously?

    overall_reasoning: str
    overall_human_preference_score: float  # Weighted combination of above scores


def evaluate_with_human_preference_rubrics(
    image_path: str, original_prompt: str, requirements: List[str]
) -> Dict[str, Any]:
    """
    Evaluate image using human preference rubrics focusing on intent matching,
    recognizability, spatial design, and user experience.

    This addresses issues like the Google logo being colored circles instead of actual letterforms.
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Create comprehensive evaluation prompt focusing on human preference
    evaluate_prompt = f"""You are evaluating an SVG image from a human preference perspective.

Original Request: {original_prompt}

Evaluate the image across these 5 key rubrics that matter to humans:

**1. INTENT MATCHING (Weight: 30%)**
Does the content actually fulfill the intended purpose? Look beyond surface requirements.
- For logos: Are they actually recognizable as the intended brand/text, not just colored shapes?
- For UI: Does it look like a functional interface users would recognize?
- For objects: Would humans identify the main subject correctly?

**2. CONTENT RECOGNIZABILITY (Weight: 25%)**
Are the key elements genuinely recognizable, not abstract representations?
- Text/logos: Can you read the actual letters/words, or are they just shapes?
- Objects: Are they clearly identifiable with proper features?
- Brands/icons: Do they match what humans would expect to see?

**3. SPATIAL DESIGN QUALITY (Weight: 20%)**
Professional layout, visual hierarchy, and design principles:
- Visual hierarchy: Do important elements stand out appropriately?
- Layout balance: Is the composition well-balanced and professional?
- Spacing and alignment: Does it follow good design principles?
- Proportions: Are elements sized appropriately relative to each other?

**4. USER EXPERIENCE (Weight: 15%)**
Would humans find this usable and appropriate?
- Functionality: For UI elements, do they look clickable/interactive?
- Clarity: Is the purpose and function immediately clear?
- Accessibility: Is text readable, elements distinguishable?
- Professional appearance: Does it meet basic quality standards?

**5. VISUAL COHERENCE (Weight: 10%)**
Do all elements work together harmoniously?
- Style consistency: Do elements match in style and quality?
- Color harmony: Do colors work well together?
- Visual flow: Does the eye move through the design naturally?

**CRITICAL: Be very strict about content that looks like abstract shapes instead of the intended content.**
For example, colored circles arranged in Google colors should score very low for intent matching and recognizability.

Original Requirements (for context):
{chr(10).join([f"{i+1}. {req}" for i, req in enumerate(requirements)])}

Respond with JSON in this exact format:
{{
    "intent_matching_score": <0.0-1.0>,
    "intent_reasoning": "<detailed explanation of how well content matches intended purpose>",
    "content_recognizability_score": <0.0-1.0>,
    "content_reasoning": "<detailed explanation of whether key elements are actually recognizable>",
    "spatial_design_score": <0.0-1.0>,
    "spatial_reasoning": "<detailed explanation of layout and design quality>",
    "user_experience_score": <0.0-1.0>,
    "ux_reasoning": "<detailed explanation of usability and appropriateness>",
    "visual_coherence_score": <0.0-1.0>,
    "coherence_reasoning": "<detailed explanation of visual harmony>",
    "overall_human_preference_score": <weighted average of above scores>,
    "overall_reasoning": "<summary of human preference evaluation>"
}}"""

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

    # Use GPT-4.1 for evaluation
    response = litellm.completion(
        model="gpt-4.1",
        messages=messages,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "HumanPreferenceResponse", "schema": HumanPreferenceResponse.model_json_schema()},
        },
    )

    # Parse response
    response_content = response.choices[0].message.content

    if not response_content or response_content.strip() == "":
        raise ValueError("Empty response from human preference evaluator")

    result = json.loads(response_content)

    # Validate the result has required fields
    required_fields = ["intent_matching_score", "content_recognizability_score", "overall_human_preference_score"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field in response: {field}")

    return result


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
    passed_threshold=0.6,  # Higher threshold for combined evaluation
    num_runs=1,
    mode="pointwise",
    max_concurrent_rollouts=50,
)
def test_svg_combined_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Combined SVG evaluation using both requirement fulfillment and human preference rubrics.

    This runs two evaluations:
    1. Original: Specific requirements per row (listwise)
    2. Human Preference: Universal rubrics for all rows (pointwise)

    Combines results to catch issues like Google logos that are just colored circles.
    """
    logger.info(f"Evaluating row {row.input_metadata.row_id} at {time.time()}")
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
        debug_dir = "svgbench_debug_combined"
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
        if not render_svg_to_png(svg_code, png_path):
            row.evaluation_result = EvaluateResult(score=0.0, reason="Failed to render SVG to PNG")
            return row

        # Run BOTH evaluations

        # 1. Original requirements-based evaluation (listwise - different per row)
        requirements_result = evaluate_with_llm_judge(png_path, requirements)
        fulfilled_count = requirements_result.get("number_of_fulfilled_requirements", 0)
        fulfilled_count = max(0, min(fulfilled_count, total_requirements))
        requirements_score = fulfilled_count / total_requirements

        # 2. Human preference evaluation (pointwise - same rubrics for all rows)
        human_pref_result = evaluate_with_human_preference_rubrics(png_path, original_prompt, requirements)
        human_pref_score = human_pref_result.get("overall_human_preference_score", 0.0)

        # Combine scores (you can adjust the weighting)
        combined_score = (requirements_score * 0.3) + (human_pref_score * 0.7)  # Emphasize human preference

        # Create comprehensive reasoning showing both evaluations
        combined_reasoning = f"""COMBINED EVALUATION (Requirements 30% + Human Preference 70%):

=== REQUIREMENTS EVALUATION (Listwise - Row-Specific) ===
Score: {requirements_score:.3f}
{requirements_result.get('reasoning', 'No reasoning provided')}

=== HUMAN PREFERENCE EVALUATION (Pointwise - Universal Rubrics) ===
Score: {human_pref_score:.3f}

🎯 Intent Matching: {human_pref_result.get('intent_matching_score', 0.0):.2f}/1.0
{human_pref_result.get('intent_reasoning', 'No reasoning provided')}

👁️ Content Recognizability: {human_pref_result.get('content_recognizability_score', 0.0):.2f}/1.0
{human_pref_result.get('content_reasoning', 'No reasoning provided')}

📐 Spatial Design Quality: {human_pref_result.get('spatial_design_score', 0.0):.2f}/1.0
{human_pref_result.get('spatial_reasoning', 'No reasoning provided')}

👤 User Experience: {human_pref_result.get('user_experience_score', 0.0):.2f}/1.0
{human_pref_result.get('ux_reasoning', 'No reasoning provided')}

🎨 Visual Coherence: {human_pref_result.get('visual_coherence_score', 0.0):.2f}/1.0
{human_pref_result.get('coherence_reasoning', 'No reasoning provided')}

{human_pref_result.get('overall_reasoning', 'No overall reasoning provided')}

=== FINAL COMBINED SCORE ===
Requirements: {requirements_score:.3f} × 30% = {requirements_score * 0.3:.3f}
Human Preference: {human_pref_score:.3f} × 70% = {human_pref_score * 0.7:.3f}
Combined: {combined_score:.3f}

The human preference evaluation helps catch issues like unrecognizable content that meets technical requirements."""

        # Store individual scores in metrics for analysis
        metrics = {
            "original_requirements_score": MetricResult(
                score=requirements_score,
                reason=f"Requirements fulfillment: {fulfilled_count}/{total_requirements} requirements met",
                is_score_valid=True,
            ),
            "overall_human_preference_score": MetricResult(
                score=human_pref_score,
                reason=human_pref_result.get("overall_reasoning", "Human preference evaluation"),
                is_score_valid=True,
            ),
            "intent_matching_score": MetricResult(
                score=human_pref_result.get("intent_matching_score", 0.0),
                reason=human_pref_result.get("intent_reasoning", "Intent matching evaluation"),
                is_score_valid=True,
            ),
            "content_recognizability_score": MetricResult(
                score=human_pref_result.get("content_recognizability_score", 0.0),
                reason=human_pref_result.get("content_reasoning", "Content recognizability evaluation"),
                is_score_valid=True,
            ),
            "spatial_design_score": MetricResult(
                score=human_pref_result.get("spatial_design_score", 0.0),
                reason=human_pref_result.get("spatial_reasoning", "Spatial design evaluation"),
                is_score_valid=True,
            ),
            "user_experience_score": MetricResult(
                score=human_pref_result.get("user_experience_score", 0.0),
                reason=human_pref_result.get("ux_reasoning", "User experience evaluation"),
                is_score_valid=True,
            ),
            "visual_coherence_score": MetricResult(
                score=human_pref_result.get("visual_coherence_score", 0.0),
                reason=human_pref_result.get("coherence_reasoning", "Visual coherence evaluation"),
                is_score_valid=True,
            ),
        }

        row.evaluation_result = EvaluateResult(score=combined_score, reason=combined_reasoning, metrics=metrics)

        return row

    except Exception as e:
        logger.error(f"Combined evaluation failed for question {row_id}: {e}")
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
