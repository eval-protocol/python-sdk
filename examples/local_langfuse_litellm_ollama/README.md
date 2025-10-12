### Local Langfuse + Fireworks Judge (optionally LiteLLM/Ollama)

This guide runs a local evaluation loop with:

- Local Langfuse via a compose file included in this repo
- Eval Protocol to pull traces and score outputs
- Fireworks-hosted LLM as the judge (accurate scoring)
- Optional: LiteLLM router in front of local backends (Ollama / llama.cpp)

References: [Langfuse Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose)

---

#### 1) Start Langfuse from the included compose file

```bash
# From repo root
docker compose -f examples/local_langfuse_litellm_ollama/langfuse-docker-compose.yml up -d
```

Export Langfuse credentials for the SDK:

```bash
export LANGFUSE_PUBLIC_KEY=local
export LANGFUSE_SECRET_KEY=local
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_ENVIRONMENT=local
```

Open the UI at `http://localhost:3000`.

---

#### 2) Launch local inference backends

Option A: Ollama

```bash
ollama serve &
ollama pull llama3.1
```

Option B: llama.cpp (OpenAI-compatible server)

```bash
# Example; adjust paths/ports/model
./server -m /path/to/Meta-Llama-3-8B-Instruct.gguf -c 8192 -ngl 33 -a 127.0.0.1 -p 8080
```

---

#### 3) Start a LiteLLM router in front of local backends

Create `litellm-config.yaml`:

```yaml
model_list:
  - model_name: "candidate/llama3.8b"
    litellm_params:
      model: "llama.cpp"
      api_base: "http://127.0.0.1:8080/v1"
      model_path: "/path/to/Meta-Llama-3-8B-Instruct.gguf"
  - model_name: "ollama/llama3.1"
    litellm_params:
      model: "ollama/llama3.1"
      api_base: "http://127.0.0.1:11434"

litellm_settings:
  drop_params: true
  telemetry: false
```

Run the router:

```bash
export LITELLM_API_KEY=local-demo-key
litellm --config litellm-config.yaml --port 4000
```

Smoke test the router:

```bash
curl -s -H "Authorization: Bearer $LITELLM_API_KEY" http://127.0.0.1:4000/v1/models | jq .
curl -s \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:4000/v1/chat/completions \
  -d '{"model":"ollama/llama3.1","messages":[{"role":"user","content":"Say hi"}]}' \
| jq -r '.choices[0].message.content'
```

---

#### 4) Seed traces into Langfuse (consolidated example)

Use the Chinook generator with PydanticAgentRolloutProcessor (no external DB required by default):

```bash
export FIREWORKS_API_KEY=...
export CHINOOK_USE_STUB_DB=1
make -C . local-generate-chinook
```

Verify adapter connectivity:

```bash
make -C . local-adapter-smoke
```

---

#### 5) Install Eval Protocol with Langfuse extras

```bash
uv pip install -e ".[langfuse]"  # or: pip install 'eval-protocol[langfuse]'
```

Ensure Fireworks credentials are set for the judge:

```bash
export FIREWORKS_API_KEY=...        # required for judge
# optional depending on your account setup
export FIREWORKS_ACCOUNT_ID=...
```

---

#### 6) Run evaluation (Fireworks-only)

```bash
export FIREWORKS_API_KEY=...
make -C . local-eval-fireworks-only
```

This pulls traces from Langfuse, runs the rollout on Fireworks, judges results on Fireworks, and pushes scores back to Langfuse.

---

#### 7) View results in Langfuse

- Open a trace and look for the evaluation score created by the run.
- Compare scores across candidate models to pick the best local model for your app.

---

#### Troubleshooting

- Langfuse not reachable: verify `LANGFUSE_HOST` and Docker health; see [Langfuse Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose)
- Judge errors: verify `FIREWORKS_API_KEY` and network access. You can switch judge model in `eval_protocol/quickstart/utils.py`.
- No results in EP UI at `http://localhost:8000`: ensure the logs server is running (`ep logs`), and that rows are being persisted under `.eval_protocol/logs.db`. With `EP_DEBUG=1`, the run prints `[EP-Debug] Logged row to EP: ...` lines.
- Ollama not being called: for direct mode, set `DIRECT_OLLAMA=1` and `OLLAMA_BASE_URL`; the run prints `[EP-Debug] DIRECT_OLLAMA=1 -> Calling Ollama: base=..., model=...`. For router mode, unset `DIRECT_OLLAMA` and confirm `LITELLM_BASE_URL` and `LITELLM_API_KEY`.
- Scores not appearing back in Langfuse: verify `FIREWORKS_API_KEY` and that the judge model can complete. With `EP_DEBUG=1`, you should see `[EP-Debug] Uploading score to Langfuse` and `Upload score success` messages.

---

#### What’s happening under the hood

- `LangfuseAdapter` pulls traces and converts them to `EvaluationRow`
- `PydanticAgentRolloutProcessor` runs the agent and logs traces
- `SingleTurnRolloutProcessor` + `aha_judge` evaluate and push scores to Langfuse
