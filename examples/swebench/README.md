SWE-bench (Remote) - Local (non-Docker) Setup and Usage

Prerequisites
- Python 3.12 environment (same one you use for this repo)
- Fireworks API key
- mini-swe-agent and datasets (for patch generation)
- SWE-bench harness installed (for evaluation)

Setup mini-swe-agent (non-Docker)
1) Install dependencies
```bash
pip install mini-swe-agent datasets
```

2) Configure API key for mini-swe-agent
```bash
mini-extra config set FIREWORKS_API_KEY <your_fireworks_key>
```

3) (Optional) Test connectivity
```bash
python3 examples/swebench/run_swe_agent_fw.py fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905 --test
```

Install SWE-bench evaluation harness
```bash
git clone https://github.com/princeton-nlp/SWE-bench
pip install -e SWE-bench
```

Environment
```bash
export FIREWORKS_API_KEY="<your_fireworks_key>"
```

Run the server
```bash
python examples/swebench/server.py
```

What the server does
- Invokes `run_swe_agent_fw.py` in batch mode with a single-slice per request
- Writes outputs to a per-row directory: `./row_{index}/`
  - `row_{index}/preds.json`
  - `row_{index}/<instance_id>/<instance_id>.traj.json`
- Runs the SWE-bench harness on `row_{index}/preds.json`

Run pytest to evaluate a model on SWE-bench
```bash
cd /Users/shrey/Documents/python-sdk
pytest examples/swebench/tests/test_swebench.py -v -s
```

Notes
- The test currently generates 10 rows by numeric index (0–9)
- Each request triggers the server to run one SWE-bench instance and write to its own `row_{index}`
- Control harness workers via: `export SWEBENCH_EVAL_WORKERS=5`
