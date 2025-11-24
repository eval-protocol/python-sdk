import asyncio
import os
from typing import List, Any

import pytest
from datasets import load_dataset
from openai import AsyncOpenAI
from dotenv import load_dotenv
from eval_protocol import SingleTurnRolloutProcessor

from eval_protocol.models import EvaluationRow, Message, EvaluateResult, MetricResult
from eval_protocol.pytest.evaluation_test import evaluation_test
from examples.calibration.evaluator import get_logprobs, calculate_ece, CLASSES, CLASS_TOKENS

load_dotenv()


# Load dataset
def load_ag_news_rows() -> List[EvaluationRow]:
    dataset = load_dataset("ag_news", split="test[:100]")
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    rows = []
    for item in dataset:
        text = item["text"]
        label_id = item["label"]
        label_name = label_map[label_id]

        row = EvaluationRow(messages=[Message(role="user", content=text)], input_metadata={"ground_truth": label_name})
        rows.append(row)
    return rows


ROWS = load_ag_news_rows()

MODELS = [
    {"model": "accounts/fireworks/models/gpt-oss-20b"},
    {"model": "accounts/fireworks/models/kimi-k2-instruct-0905"},
    {"model": "accounts/fireworks/models/kimi-k2-thinking"},
]

# Save rows to JSONL
import json

jsonl_path = "examples/calibration/ag_news_subset.jsonl"
with open(jsonl_path, "w") as f:
    for row in ROWS:
        f.write(row.model_dump_json() + "\n")


@evaluation_test(
    mode="all",  # Batch mode
    input_dataset=[jsonl_path],
    completion_params=MODELS,
    rollout_processor=SingleTurnRolloutProcessor(),
    max_dataset_rows=5,
)
async def test_calibration(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    # completion_params is passed for the current run
    completion_params = rows[0].input_metadata.completion_params
    model_id = completion_params["model"]
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not found")

    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )

    print(f"\nEvaluating model: {model_id}")

    tasks = []
    for row in rows:
        prompt = f"{row.messages[-1].content}\n\nClassify the above text into one of: {', '.join(CLASSES)}.\nAnswer:"
        tasks.append(get_logprobs(client, model_id, prompt, CLASS_TOKENS))

    probs_list = await asyncio.gather(*tasks)

    all_confs = []
    all_preds = []
    all_labels = []

    for i, probs in enumerate(probs_list):
        row = rows[i]
        gt_class = row.get_input_metadata("ground_truth")

        pred_class = max(probs, key=probs.get)
        conf = probs[pred_class]

        all_confs.append(conf)
        all_preds.append(pred_class)
        all_labels.append(gt_class)

        brier = 0.0
        for cls in CLASSES:
            target = 1.0 if cls == gt_class else 0.0
            prob = probs.get(cls, 0.0)
            brier += (prob - target) ** 2

        row.evaluation_result = EvaluateResult(
            score=brier,
            reason=f"Predicted: {pred_class} ({conf:.2f}), GT: {gt_class}. Brier: {brier:.4f}",
            metrics={
                "confidence": MetricResult(score=conf, reason="Model confidence"),
                "correct": MetricResult(score=1.0 if pred_class == gt_class else 0.0, reason="Accuracy"),
            },
        )

    # Calculate global ECE
    class_to_int = {c: i for i, c in enumerate(CLASSES)}
    pred_ints = [class_to_int.get(p, -1) for p in all_preds]
    label_ints = [class_to_int.get(l, -1) for l in all_labels]

    ece = calculate_ece(pred_ints, all_confs, label_ints)

    # Calculate avg accuracy and brier for logging
    avg_acc = sum([1.0 if p == l else 0.0 for p, l in zip(all_preds, all_labels)]) / len(all_preds)
    avg_brier = sum([r.evaluation_result.score for r in rows]) / len(rows)

    print(f"Results for {model_id}:")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  Avg Brier Score: {avg_brier:.4f}")
    print(f"  ECE: {ece:.4f}")

    # Attach ECE to all rows (or just log it)
    for row in rows:
        row.evaluation_result.metrics["global_ece"] = MetricResult(
            score=ece, reason="Global Expected Calibration Error"
        )

    return rows
