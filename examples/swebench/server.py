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
rollout_states = {}

@app.post("/init")
def init(req: InitRequest):
    # Allow Eval Protocol to dynamically configure ES endpoint
    if req.elastic_search_config:
        handler.configure(req.elastic_search_config)

    # Tag all logs for this rollout_id
    logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    rollout_states[req.metadata.rollout_id] = {
        "terminated": False,
        "status": "running",
        "instance_id": req.metadata.row_id,
    }

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
            env["PYTHONPATH"] = "/Users/shrey/Documents/python-sdk/examples/swebench:" + env.get("PYTHONPATH", "")

            # Determine output directory (from env or default)
            out_dir = os.getcwd()

            from pathlib import Path
            
            script_path = str((Path(__file__).parent / "run_swe_agent_fw.py").resolve())
            
            # Extract model_kwargs from req.metadata (forwarded from input_metadata)
            model_kwargs = {}
            logger.info(f"DEBUG: req.metadata attributes: {dir(req.metadata)}")
            if hasattr(req.metadata, "model_kwargs"):
                mk = getattr(req.metadata, "model_kwargs", None)
                logger.info(f"DEBUG: Found req.metadata.model_kwargs = {mk}")
                if isinstance(mk, dict):
                    model_kwargs = mk
                    logger.info(f"Extracted model_kwargs from metadata: {model_kwargs}")
            else:
                logger.info(f"DEBUG: req.metadata has NO model_kwargs attribute")
            
            # Set tracing URL
            if req.model_base_url:
                env["TRACING_BASE_URL"] = req.model_base_url

            cmd = [
                "python3",
                script_path,
                req.model,
                "--single", str(single_index),
                "--exit-immediately",
                "--output", str(out_dir),
                "--model-class", "tracing_model.TracingFireworksModel",
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
            row_dir = Path(out_dir) / f"row_{single_index}"
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


            # Stream stdout/stderr to logs
            # assert proc.stdout is not None and proc.stderr is not None
            # for line in proc.stdout:
            #     logger.info(line.rstrip("\n"))
            # for line in proc.stderr:
            #     logger.warning(line.rstrip("\n"))

            # ret = proc.wait()
            # logger.info(f"mini-swe-agent exited with code {ret}")

            # Use row-specific preds.json to avoid cross-run interference
            preds_path = row_dir / "preds.json"
            if preds_path.exists():
                logger.info(f"Using preds.json at: {preds_path}")
            else:
                logger.error(f"No preds.json found at {preds_path}")

            # 2) Run SWE-bench evaluation harness on preds.json
            preds_path_str = str(preds_path)
            eval_cmd = [
                "python3", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", "princeton-nlp/SWE-bench_Verified",
                "--predictions_path", preds_path_str,
                "--max_workers", str(os.getenv("SWEBENCH_EVAL_WORKERS", "5")),
                "--run_id", "eval-run",
            ]
            logger.info("Starting SWE-bench harness: %s", " ".join(map(str, eval_cmd)))
            eval_proc = subprocess.Popen(
                eval_cmd, cwd=str(row_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            assert eval_proc.stdout is not None
            for line in eval_proc.stdout:
                logger.info(line.rstrip("\n"))
            eval_rc = eval_proc.wait()
            # logger.info(f"SWE-bench harness exited with code {eval_rc}")

        except Exception as e:
            # Best-effort: mark error but still finish to unblock polling
            logger.error(f"Rollout error: {e}", extra={"status": Status.rollout_error(str(e))})
        finally:
            # Always mark finished so RemoteRolloutProcessor stops polling
            logger.info("Rollout completed", extra={"status": Status.rollout_finished()})

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "accepted"}

@app.get("/status")
def status(rollout_id: str):
    return rollout_states.get(rollout_id, {"terminated": False})

def main():
    host = os.getenv("REMOTE_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("REMOTE_SERVER_PORT", "3000"))
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
