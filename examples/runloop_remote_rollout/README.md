# Runloop Remote Rollout Example

This example hosts an Eval Protocol remote rollout server in a Runloop Devbox.

## Requirements

```bash
pip install "eval-protocol[runloop]"
export RUNLOOP_API_KEY=...
export FIREWORKS_API_KEY=...
```

Create a Runloop blueprint that contains this repository and its Python dependencies, then set `RUNLOOP_BLUEPRINT_ID`:

```bash
export RUNLOOP_BLUEPRINT_ID=bpt_your_blueprint_id
pytest examples/runloop_remote_rollout/test_eval.py
```

The processor starts:

```bash
python -m uvicorn examples.runloop_remote_rollout.server:app --host 0.0.0.0 --port 8000
```

The server receives `POST /init`, performs a chat completion through the Fireworks tracing base URL provided by Eval Protocol, and logs rollout completion using Fireworks tracing metadata.
