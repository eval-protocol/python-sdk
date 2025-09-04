import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    Message,
    MetricResult,
    ChatCompletionContentPartTextParam,
)
from eval_protocol.pytest.default_single_turn_rollout_process import (
    SingleTurnRolloutProcessor,
)
from eval_protocol.pytest.evaluation_test import evaluation_test
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

# -------------------------
# Lightweight ports of LiveBench scoring utilities for data_analysis tasks
# -------------------------


def _lb_clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w]", "", text)
    return text


def _extract_last_boxed_segment(text: str) -> Optional[str]:
    # Extract the last occurrence of \\boxed{...} or \\framebox{...}
    pattern = r"\\(?:boxed|framebox)\{(.*?)\}"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return matches[-1]


def _coerce_content_to_str(content: str | list[ChatCompletionContentPartTextParam] | None) -> str:
    if isinstance(content, list):
        return "".join([getattr(p, "text", str(p)) for p in content])
    return str(content or "")


def _cta_process_results(ground_truth: str, llm_answer: str) -> int:
    parsed_answer = llm_answer
    if "\\boxed{" in parsed_answer or "\\framebox{" in parsed_answer:
        boxed = _extract_last_boxed_segment(parsed_answer)
        if boxed is not None:
            parsed_answer = boxed
        parsed_answer = parsed_answer.replace("\\text{", "").replace("}", "").replace("\\", "")

    gt_clean = _lb_clean_text(ground_truth)
    ans_clean = _lb_clean_text(parsed_answer)
    if gt_clean == ans_clean:
        return 1
    # Suffix match to handle answers like "... answer: XYZ"
    if len(ans_clean) >= len(gt_clean) and ans_clean[-len(gt_clean) :] == gt_clean:
        return 1
    return 0


def _tj_clean_llm_output(s: str) -> Dict[str, Any]:
    # Try to extract the last <solution>...</solution>
    m = re.findall(r"<solution>(.*?)</solution>", s, re.DOTALL)
    if len(m) > 0:
        return _tj_clean_llm_output(m[-1].strip())

    candidate: Optional[str] = None
    # Prefer code blocks (python/json/any)
    for fence in ("```python", "```json", "```"):
        mm = re.findall(r"%s(.*?)```" % re.escape(fence), s.replace("\n", ""), re.MULTILINE)
        if mm:
            candidate = mm[-1]
            break
    # Fallback to boxed
    if candidate is None and "\\boxed" in s:
        boxed = _extract_last_boxed_segment(s.replace("\n", ""))
        if boxed:
            # Convert \text{"str"} to 'str' and strip backslashes
            candidate = re.sub(r"\\text{['\"](.*?)['\"]}", r"'\1'", boxed).replace("\\", "")
    if candidate is None:
        candidate = s

    # Make JSON-like to python literal
    candidate = candidate.replace("null", "None")
    try:
        from ast import literal_eval

        parsed = literal_eval(candidate)
        if not isinstance(parsed, dict):
            return {}
        # Drop None values
        for k in list(parsed.keys()):
            if parsed[k] is None:
                del parsed[k]
        return parsed
    except Exception:
        return {}


def _tablejoin_process_results(ground_truth: Any, llm_answer: str) -> float:
    import json as _json
    from ast import literal_eval

    # Parse GT into dict if needed
    gt: Dict[str, Any]
    if isinstance(ground_truth, str):
        try:
            gt = literal_eval(ground_truth)
        except Exception:
            try:
                gt = _json.loads(ground_truth)
            except Exception:
                return 0.0
    else:
        gt = dict(ground_truth)

    pred = _tj_clean_llm_output(llm_answer)
    if len(pred) == 0:
        return 0.0

    tp = 0
    fp = 0
    fn = 0
    for k, v in pred.items():
        gt_v = gt.get(k, None)
        if gt_v is None:
            fp += 1
        elif gt_v == v:
            tp += 1
        else:
            fp += 1
            fn += 1
    for k, v in gt.items():
        if k not in pred:
            fn += 1
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    # Round to 2 decimals to mirror LiveBench
    return round((2 * tp) / denom, 2)


def _tablereformat_process_results(input_command: str, ground_truth: str, llm_answer: str, version: str) -> int:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return 0

    import math as _math
    import traceback as _traceback
    from io import StringIO

    def _read_df_v1(df_type: str, df_str: str):
        if df_type == "json":
            for orient in ("index", "records", "records", "table", "values"):
                try:
                    return pd.read_json(StringIO(df_str), orient=orient)
                except Exception:
                    pass
            return pd.read_json(StringIO(df_str), orient="values")
        if df_type == "jsonl":
            return pd.read_json(StringIO(df_str), orient="records", lines=True)
        if df_type == "html":
            return pd.concat(pd.read_html(StringIO(df_str)), axis=0)
        if df_type == "csv":
            return pd.read_csv(StringIO(df_str))
        if df_type == "markdown":
            return pd.read_table(StringIO(df_str), sep="|", header=0, index_col=1, skipinitialspace=True)
        if df_type == "tsv":
            return pd.read_csv(StringIO(df_str), sep="\t")
        raise ValueError(f"Unsupported type {df_type}")

    def _read_df_v2(df_type: str, df_str: str):
        if df_type == "json":
            for orient in ("table", "index", "records"):
                try:
                    return pd.read_json(StringIO(df_str), orient=orient)
                except Exception:
                    pass
            return None
        if df_type == "jsonl":
            return pd.read_json(StringIO(df_str), orient="records", lines=True)
        if df_type == "html":
            return pd.concat(pd.read_html(StringIO(df_str)), axis=0)
        if df_type == "csv":
            return pd.read_csv(StringIO(df_str))
        if df_type == "markdown":
            # Remove alignment line
            lines = df_str.strip().split("\n")
            header = lines[0]
            data_lines = lines[2:] if len(lines) > 2 else []
            processed = header + "\n" + "\n".join(data_lines)
            df = pd.read_table(StringIO(processed), sep="|", header=0, skipinitialspace=True).iloc[:, 1:-1]
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.strip()
            return df
        if df_type == "tsv":
            return pd.read_csv(StringIO(df_str), sep="\t")
        raise ValueError(f"Unsupported type {df_type}")

    def _clean_llm_output(s: str) -> str:
        m = re.findall(r"```json\n(.*?)```", s, re.DOTALL)
        if m:
            return m[-1].strip()
        m = re.findall(r"```html\n(.*?)```", s, re.DOTALL)
        if m:
            return m[-1].strip()
        s = re.sub(r"^```.*\n", "", s)
        s = s.replace("&amp;", "&")
        return s.replace("```", "").strip()

    def _remove_initial_phrase(text: str) -> str:
        return re.sub(r"^\s*(Here|Input)\b.*?\b(format|table)\s*[:)]\s*", "", text, flags=re.IGNORECASE).strip()

    def _read_sep_table_from_text(text: str, header: str, sep: str):
        text = text.strip()
        lines = text.split("\n")
        header_line = 0
        while header_line < len(lines) and lines[header_line].strip() != header.strip():
            header_line += 1
        if header_line == len(lines) or lines[header_line].strip() != header.strip():
            return None
        table = lines[header_line:]
        parsed = None
        while parsed is None and table:
            try:
                parsed = pd.read_csv(StringIO("\n".join(table)), sep=sep)
            except Exception:
                table = table[:-1]
        return parsed

    def _read_jsonl_table_from_text(text: str, header_cols: List[str]):
        rows = []
        for line in text.strip().split("\n"):
            if len(line) < 2 or line[0] != "{" or line[-1] != "}":
                continue
            if not all(col in line for col in header_cols):
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if not rows:
            return None
        import pandas as _pd

        return _pd.DataFrame(rows)

    # Determine formats from the instruction
    if version == "v1":
        input_fmt = input_command.split("Please convert the Input Table from ")[1].split(" format")[0].lower()
        output_fmt = (
            input_command.split("Please convert the Input Table from ")[1]
            .split("format to ")[1]
            .split(" format")[0]
            .lower()
        )
    else:
        lines = input_command.split("\n")
        input_fmt = (
            [line for line in lines if "Source Format" in line][-1].split("Source Format: ")[-1].strip().lower()
        )
        output_fmt = (
            [line for line in lines if "Target Format" in line][-1].split("Target Format: ")[-1].strip().lower()
        )

    reader = _read_df_v1 if version == "v1" else _read_df_v2
    gt_df = reader(output_fmt, ground_truth)
    assert gt_df is not None, "GT dataframe is None"

    llm_clean = _clean_llm_output(llm_answer)
    llm_clean = _remove_initial_phrase(llm_clean)
    try:
        llm_df = reader(output_fmt, llm_clean)
    except Exception:
        llm_df = None
        if output_fmt in ("csv", "tsv") and gt_df is not None:
            header = (",", "\t")[output_fmt == "tsv"].join(list(gt_df.columns))
            llm_df = _read_sep_table_from_text(llm_clean, header, sep="," if output_fmt == "csv" else "\t")
        elif output_fmt == "jsonl" and gt_df is not None:
            llm_df = _read_jsonl_table_from_text(llm_clean, list(gt_df.columns))
        if llm_df is None:
            return 0

    # Compare
    assert llm_df is not None, "LLM dataframe is None"
    assert gt_df is not None, "GT dataframe is None"
    try:
        gt_df.columns = [str(s).strip() for s in gt_df.columns]
        if "index" in gt_df.columns:
            gt_df = gt_df.drop(columns=["index"])
        llm_df.columns = [str(s).strip() for s in llm_df.columns]
        if "index" in llm_df.columns:
            llm_df = llm_df.drop(columns=["index"])
        assert len(llm_df) == len(gt_df)
        assert sorted(llm_df.columns) == sorted(gt_df.columns)
        for i in range(len(llm_df)):
            for key in llm_df.columns:
                lv = llm_df.iloc[i][key]
                gv = gt_df.iloc[i][key]
                if isinstance(lv, str):
                    lv = lv.strip()
                if isinstance(gv, str):
                    gv = gv.strip()
                # Numeric tolerance for floats
                try:
                    lvf = float(lv)
                    gvf = float(gv)
                    if _math.isnan(lvf) and _math.isnan(gvf):
                        continue
                    assert abs(lvf - gvf) < 1e-6
                except Exception:
                    assert str(lv) == str(gv)
    except AssertionError:
        return 0
    except Exception:
        # Silent on failure, match LiveBench robustness
        _traceback.print_exc()
        return 0
    return 1


# -------------------------
# Custom Rollout Processor to preserve ground truth
# -------------------------


class LiveBenchGroundTruthRolloutProcessor(RolloutProcessor):
    """Rollout processor that preserves ground truth data from pre-loaded datasets."""

    def __init__(self, task_rows: List[EvaluationRow]):
        super().__init__()
        self.single_turn_processor = SingleTurnRolloutProcessor()
        # Create a mapping from message content to ground truth
        self.ground_truth_map = {}
        for row in task_rows:
            if row.messages and len(row.messages) >= 2:  # system + user messages
                user_msg = row.messages[1].content  # user message is typically second
                if user_msg:
                    self.ground_truth_map[str(user_msg)] = row.ground_truth

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Set ground truth on rows based on message content, then delegate to SingleTurnRolloutProcessor."""
        processed: List[EvaluationRow] = []

        for row in rows:
            # Find matching ground truth based on user message content
            if row.messages and len(row.messages) >= 2:
                user_msg = row.messages[1].content  # user message
                if user_msg and str(user_msg) in self.ground_truth_map:
                    row.ground_truth = self.ground_truth_map[str(user_msg)]
            processed.append(row)

        # Delegate to SingleTurnRolloutProcessor
        return self.single_turn_processor(processed, config)


# -------------------------
# Dataset loading from Hugging Face at import time
# -------------------------

SYSTEM_PROMPT = "You are a helpful data analyst. Read the task and answer precisely."


def _load_livebench_da_messages(task_name: str) -> List[EvaluationRow]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required for LiveBench Data Analysis benchmarks. Please 'pip install datasets'."
        ) from e

    ds = load_dataset("livebench/data_analysis", split="test")
    rows: List[EvaluationRow] = []
    for ex in ds:
        if str(ex.get("task", "")) != task_name:
            continue
        question_text = str(ex.get("turns", [""])[0])
        ground_truth = ex.get("ground_truth")
        release = ex.get("livebench_release_date", "")
        try:
            gt_payload = json.dumps({"ground_truth": ground_truth, "release": release}, ensure_ascii=False)
        except TypeError:
            gt_payload = json.dumps({"ground_truth": str(ground_truth), "release": str(release)})
        rows.append(
            EvaluationRow(
                messages=[
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=question_text),
                ],
                ground_truth=gt_payload,
            )
        )
    if not rows:
        raise RuntimeError(f"No rows found for LiveBench data_analysis task '{task_name}'")
    return rows


def _extract_gt(row: EvaluationRow) -> Dict[str, Any]:
    # For LiveBench Data Analysis, we fetch the ground truth from the HF dataset
    # and store it in the top-level ground_truth field in the adapter below.
    # Here, just parse row.ground_truth if it contains a JSON payload, else string.
    if row.ground_truth is None:
        return {"ground_truth": None, "release": None}
    try:
        payload = json.loads(str(row.ground_truth))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {"ground_truth": row.ground_truth, "release": None}


# -------------------------
# CTA
# -------------------------

_CTA_ROWS = _load_livebench_da_messages("cta")


@evaluation_test(
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    # Wrap dataset messages in an extra list to match Sequence[list[InputMessagesParam]]
    input_messages=[[[m for m in r.messages] for r in _CTA_ROWS]],
    rollout_processor_kwargs={"extra_body": {"reasoning_effort": "low"}},
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=None,
    num_runs=4,
    mode="pointwise",
)
def test_livebench_cta_pointwise(row: EvaluationRow) -> EvaluationRow:
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    raw_content = assistant_msgs[-1].content if assistant_msgs else ""
    content = _coerce_content_to_str(raw_content)
    payload = _extract_gt(row)
    gt = payload.get("ground_truth")
    gt_str = str(gt) if gt is not None else ""

    score_val = float(_cta_process_results(gt_str, content or "")) if gt_str else 0.0
    is_valid = bool(gt_str)

    row.evaluation_result = EvaluateResult(
        score=score_val,
        reason=("Matched" if score_val == 1.0 else "Not matched"),
        is_score_valid=is_valid,
        metrics={
            "exact_match": MetricResult(
                score=score_val,
                is_score_valid=is_valid,
                reason=("Exact/suffix match" if score_val == 1.0 else "Mismatch"),
            )
        },
    )
    return row


# -------------------------
# Table Join
# -------------------------

_TABLEJOIN_ROWS = _load_livebench_da_messages("tablejoin")


@evaluation_test(
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    input_messages=[[[m for m in r.messages] for r in _TABLEJOIN_ROWS]],
    rollout_processor_kwargs={"extra_body": {"reasoning_effort": "low"}},
    rollout_processor=LiveBenchGroundTruthRolloutProcessor(_TABLEJOIN_ROWS),
    aggregation_method="mean",
    passed_threshold=None,
    num_runs=4,
    mode="pointwise",
)
def test_livebench_tablejoin_pointwise(row: EvaluationRow) -> EvaluationRow:
    user_msgs = [m for m in row.messages if m.role == "user"]
    question = _coerce_content_to_str(user_msgs[-1].content if user_msgs else "")
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    content = _coerce_content_to_str(assistant_msgs[-1].content if assistant_msgs else "")
    payload = _extract_gt(row)
    gt = payload.get("ground_truth")

    score_val = float(_tablejoin_process_results(gt, content or ""))
    is_valid = True

    row.evaluation_result = EvaluateResult(
        score=score_val,
        reason=f"F1 score: {score_val:.2f}",
        is_score_valid=is_valid,
        metrics={
            "f1": MetricResult(
                score=score_val,
                is_score_valid=is_valid,
                reason="Entity/relation mapping F1",
            )
        },
    )
    return row


# -------------------------
# Table Reformat
# -------------------------

_TABLEREFORMAT_ROWS = _load_livebench_da_messages("tablereformat")


@evaluation_test(
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    input_messages=[[[m for m in r.messages] for r in _TABLEREFORMAT_ROWS]],
    rollout_processor_kwargs={"extra_body": {"reasoning_effort": "low"}},
    rollout_processor=LiveBenchGroundTruthRolloutProcessor(_TABLEREFORMAT_ROWS),
    aggregation_method="mean",
    passed_threshold=None,
    num_runs=4,
    mode="pointwise",
)
def test_livebench_tablereformat_pointwise(row: EvaluationRow) -> EvaluationRow:
    user_msgs = [m for m in row.messages if m.role == "user"]
    question = _coerce_content_to_str(user_msgs[-1].content if user_msgs else "")
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    content = _coerce_content_to_str(assistant_msgs[-1].content if assistant_msgs else "")
    payload = _extract_gt(row)
    gt = payload.get("ground_truth")
    release = payload.get("release") or ""
    version = "v2" if str(release) >= "2025-04-25" else "v1"

    gt_str = str(gt) if gt is not None else ""
    score_int = _tablereformat_process_results(question or "", gt_str, content or "", version)
    score_val = float(score_int)
    is_valid = bool(gt_str)

    row.evaluation_result = EvaluateResult(
        score=score_val,
        reason=("Table matches" if score_val == 1.0 else "Table mismatch"),
        is_score_valid=is_valid,
        metrics={
            "structure_exact": MetricResult(
                score=score_val,
                is_score_valid=is_valid,
                reason="Exact structure and values match",
            )
        },
    )
    return row
