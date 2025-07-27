# Proposal: Pytest-Based Evaluation Workflow

This document outlines a workflow for authoring evaluations as pytest tests and
aggregating their results for RL fine-tuning (RFT) or supervised fine-tuning
(SFT).

## 1. Evaluation Decorator

* Provide a new `evaluation_test` decorator in `eval_protocol.pytest_utils`.
* Accept optional arguments:
  * `samples`: path to a JSONL dataset or HF dataset name
  * `batch`: whether to call the wrapped reward function in batch mode
  * `max_samples`: limit the number of samples loaded
  * `env_url`: optional URL for an RL environment which is started and torn down
    around the test
* The decorator internally wraps the user function with `reward_function` to
  validate that it returns an `EvaluateResult`.
* During the test, data is loaded and each sample is evaluated. Any invalid score
  or exceptions will cause the test to fail.

## 2. RL Environment Support

When `env_url` is provided, the decorator performs setup/teardown by launching
and polling the environment URL. Step data from the environment can be attached
via `EvaluateResult.step_outputs` as usual. This allows multi-turn RL scenarios
in regular pytest tests.

## 3. CI/CD Workflow

1. Developers write evaluation tests using `@evaluation_test`.
2. CI runs `pytest` to execute all evaluations whenever the system prompt or
   reward code changes.
3. A pytest plugin collects `EvaluateResult` objects from all tests and writes a
   single JSON file summarizing scores and metrics.
4. This file can be used to detect regressions (single aggregated score or
   per-test scores) and also serves as input for SFT/RFT pipelines.

The same serialized results together with the input datasets and reward code can
be uploaded for controlled rollouts or fine-tuning. This unifies local
validation, CI checks, and training data generation.
