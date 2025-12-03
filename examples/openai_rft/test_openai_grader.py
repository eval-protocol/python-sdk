import os
import requests

from eval_protocol.integrations.openai_rft import build_python_grader_from_evaluation_test
from examples.openai_rft.example_rapidfuzz import rapidfuzz_eval


api_key = os.environ["OPENAI_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}"}

grader = build_python_grader_from_evaluation_test(rapidfuzz_eval)  # {"type": "python", "source": "..."}

# validate the grader
resp = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/validate",
    json={"grader": grader},
    headers=headers,
)
print("validate response:", resp.text)

# run the grader once with a dummy item/sample
payload = {
    "grader": grader,
    "item": {"reference_answer": "fuzzy wuzzy had no hair"},
    "model_sample": "fuzzy wuzzy was a bear",
}
resp = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=payload,
    headers=headers,
)
print("run response:", resp.text)
