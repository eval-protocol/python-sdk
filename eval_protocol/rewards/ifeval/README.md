# IFEval Reward Function

Evaluates how well model responses follow instruction constraints. Returns a partial credit score (0.0 to 1.0).

## Quick Start

```python
import sys
sys.path.insert(0, '/path/to/eval_protocol/rewards/ifeval')
from reward import ifeval_partial_credit_reward

response = "Hello world! This is my response."
ground_truth = {
    "instruction_id": ["keywords:existence"],
    "kwargs": [{"keywords": ["hello", "world"]}]
}

score = ifeval_partial_credit_reward(response, ground_truth)
# Score: 1.0 (all constraints satisfied)
```

## Dependencies

```bash
pip install spacy nltk langdetect emoji syllapy immutabledict
python -m spacy download en_core_web_sm
```

## Notes

- Automatically strips `<think>...</think>` tags before evaluation
- Ground truth can be a dict, list, or JSON string
- 112 total constraints (54 IFEval/IFTrain + 58 IFBench OOD)

## File Sources

**Copied from `open-instruct/open_instruct/IFEvalG/`:**
- `ifeval_instructions.py` (from `instructions.py`)
- `ifeval_registry.py` (from `instructions_registry.py`)
- `ifeval_util.py` (from `instructions_util.py`)

**Copied from `IFBench/`:**
- `ifbench_instructions.py` (from `instructions.py`)
- `ifbench_registry.py` (from `instructions_registry.py`)
- `ifbench_util.py` (from `instructions_util.py`)

**New code:**
- `reward.py` - main reward function
- `__init__.py` - package exports
