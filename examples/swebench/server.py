"""Minimal SWE-bench server - wraps run_swe_agent_fw.py with tracing via model_base_url."""

import os
import threading
import subprocess
import logging
from fastapi import FastAPI
import uvicorn

from eval_protocol import Status, InitRequest, ElasticsearchDirectHttpHandler, RolloutIdFilter

app = FastAPI()

# Attach Elasticsearch handler to root logger (Eval Protocol UI)
handler = ElasticsearchDirectHttpHandler()
logging.getLogger().addHandler(handler)
# rollout_states = {}


@app.post("/init")
def init(req: InitRequest):
    # Allow Eval Protocol to dynamically configure ES endpoint
    if req.elastic_search_config:
        handler.configure(req.elastic_search_config)

    # Tag all logs for this rollout_id
    logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    # rollout_states[req.metadata.rollout_id] = {
    #     "terminated": False,
    #     "status": "running",
    #     "instance_id": req.metadata.row_id,
    # }

    def _worker():
        try:
            # Validate model
            if not req.model:
                raise ValueError("model is required")

            if not req.metadata or not req.metadata.row_id:
                raise ValueError("metadata.row_id is required and must be an integer index as string, e.g. '0'")
            try:
                single_index = int(str(req.metadata.row_id))
            except ValueError:
                raise ValueError(f"row_id must be an integer index for --single, got: {req.metadata.row_id}")
            env = os.environ.copy()
            # Build environment for subprocess
            if "FIREWORKS_API_KEY" in os.environ:
                env["FIREWORKS_API_KEY"] = os.environ["FIREWORKS_API_KEY"]
            # Make sure the tracing model module is importable by the subprocess
            # so "tracing_model.TracingFireworksModel" can be imported
            from pathlib import Path

            script_dir = Path(__file__).parent
            env["PYTHONPATH"] = f"{script_dir}:{env.get('PYTHONPATH', '')}"

            # Sandbox by invocation_id to isolate concurrent test runs
            from pathlib import Path

            invocation_id = req.metadata.invocation_id
            base_dir = Path(os.getcwd()) / invocation_id
            base_dir.mkdir(parents=True, exist_ok=True)

            script_path = str((Path(__file__).parent / "run_swe_agent_fw.py").resolve())

            # Extract model_kwargs from req.metadata (forwarded from input_metadata)
            model_kwargs = {}
            # convert to logger.debug everywhere, remove debug then
            logger.debug(f"req.metadata attributes: {dir(req.metadata)}")

            if hasattr(req.metadata, "model_kwargs"):
                mk = getattr(req.metadata, "model_kwargs", None)
                logger.debug(f"Found req.metadata.model_kwargs = {mk}")
                if isinstance(mk, dict):
                    model_kwargs = mk
                    logger.debug(f"Extracted model_kwargs from metadata: {model_kwargs}")
            else:
                logger.debug("req.metadata has NO model_kwargs attribute")

            # Set tracing URL
            if req.model_base_url:
                env["TRACING_BASE_URL"] = req.model_base_url

            cmd = [
                "python3",
                script_path,
                req.model,
                "--single",
                str(single_index),
                "--exit-immediately",
                "--output",
                str(base_dir),
                "--model-class",
                "tracing_model.TracingFireworksModel",
            ]
            # Forward model kwargs as CLI flags to the wrapper
            if model_kwargs.get("reasoning") in ("low", "medium", "high"):
                cmd.extend(["--reasoning", str(model_kwargs["reasoning"])])
            if model_kwargs.get("temperature") is not None:
                cmd.extend(["--temperature", str(model_kwargs["temperature"])])
            if model_kwargs.get("max_tokens") is not None:
                cmd.extend(["--max-tokens", str(model_kwargs["max_tokens"])])
            import json

            # Log path inside row directory for this run
            row_dir = base_dir / f"row_{single_index}"
            row_dir.mkdir(parents=True, exist_ok=True)
            log_path = row_dir / f"agent_{single_index}.log"

            # Run without streaming; write all output to a log file; wait until completion
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                ret = proc.wait()

            # Use row-specific preds.json to avoid cross-run interference
            preds_path = row_dir / "preds.json"
            if preds_path.exists():
                logger.info(f"Using preds.json at: {preds_path}")
            else:
                logger.error(f"No preds.json found at {preds_path}")

            # 2) Run SWE-bench evaluation harness on preds.json
            preds_path_str = str(preds_path)
            eval_cmd = [
                "python3",
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                "princeton-nlp/SWE-bench_Verified",
                "--predictions_path",
                preds_path_str,
                "--max_workers",
                str(os.getenv("SWEBENCH_EVAL_WORKERS", "5")),
                "--run_id",
                "eval-run",
            ]
            logger.info("Starting SWE-bench harness: %s", " ".join(map(str, eval_cmd)))
            eval_proc = subprocess.Popen(
                eval_cmd, cwd=str(row_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            assert eval_proc.stdout is not None
            for line in eval_proc.stdout:
                logger.info(line.rstrip("\n"))
            eval_rc = eval_proc.wait()

            # Collect evaluation results to send via Elasticsearch
            import yaml

            instance_id = None
            resolved = None
            exit_reason = None

            if preds_path.exists():
                try:
                    preds = json.loads(preds_path.read_text())
                    instance_id = next(iter(preds.keys()), None)
                except Exception:
                    pass

            if instance_id:
                model_id = req.model
                if model_id:
                    safe_model = model_id.replace("/", "__").replace(":", "-")
                    report_path = (
                        row_dir / "logs" / "run_evaluation" / "eval-run" / safe_model / instance_id / "report.json"
                    )

                    if report_path.exists():
                        try:
                            report_data = json.loads(report_path.read_text())
                            resolved = bool(report_data.get(instance_id, {}).get("resolved", False))
                        except Exception:
                            pass

                    if resolved is None:
                        exit_files = sorted(row_dir.glob("exit_statuses_*.yaml"))
                        if exit_files:
                            try:
                                status_doc = yaml.safe_load(exit_files[-1].read_text()) or {}
                                by_status = status_doc.get("instances_by_exit_status", {})
                                for status_name, ids in by_status.items():
                                    if instance_id in (ids or []):
                                        resolved = False
                                        exit_reason = status_name
                                        break
                            except Exception:
                                pass

            results_data = {
                "instance_id": instance_id,
                "resolved": resolved,
                "exit_reason": exit_reason,
                "row_id": str(single_index),
            }

        except Exception as e:
            # Best-effort: mark error but still finish to unblock polling
            results_data = {"error": str(e), "row_id": str(single_index)}
            logger.error(f"Rollout error: {e}", extra={"status": Status.rollout_error(str(e))})
        finally:
            # Create and log EvaluateResult in standardized format
            from eval_protocol.models import EvaluateResult, MetricResult

            if resolved is not None:
                reason = f"instance={instance_id}, resolved={resolved}"
                if exit_reason:
                    reason += f", exit_reason={exit_reason}"

                eval_result = EvaluateResult(
                    score=1.0 if resolved else 0.0,
                    reason=reason,
                    is_score_valid=True,
                    metrics={
                        "resolved": MetricResult(
                            score=1.0 if resolved else 0.0,
                            is_score_valid=True,
                            reason=f"resolved={resolved}",
                            value=int(resolved),
                        )
                    },
                )
                logger.info(
                    f"EVAL_RESULT:{eval_result.model_dump_json()}", extra={"status": Status.rollout_finished()}
                )
            else:
                logger.info("EVAL_RESULT:null", extra={"status": Status.rollout_finished()})

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "accepted"}


# @app.get("/status")
# def status(rollout_id: str):
#     return rollout_states.get(rollout_id, {"terminated": False})


def main():
    host = os.getenv("REMOTE_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("REMOTE_SERVER_PORT", "3000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
