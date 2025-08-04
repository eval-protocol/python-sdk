# Frozen Lake Agent Evaluation

This example demonstrates LLM agent evaluation on the Frozen Lake game using eval-protocol's HTTP rollout framework. The agent must navigate from start (S) to goal (G) while avoiding holes (H).

## Quick Start

### Setup
```bash
# For Fireworks AI
export FIREWORKS_API_KEY="your_fireworks_api_key"
export MODEL_AGENT="fireworks/accounts/fireworks/models/qwen3-235b-a22b"

# For OpenAI
export OPENAI_API_KEY="your_openai_api_key"
export MODEL_AGENT="openai/gpt-4o-mini"

# For other providers, set appropriate API key and MODEL_AGENT
```

### Run Evaluation
```bash
# Batch evaluation (8 parallel rollouts) - recommended
eval-protocol agent-eval --task-def examples/frozen_lake/client/task_def.yaml

# Single rollout for debugging
eval-protocol agent-eval --task-def examples/frozen_lake/client/task_def.yaml --num-rollouts 1

# Custom batch size
eval-protocol agent-eval --task-def examples/frozen_lake/client/task_def.yaml --num-rollouts 16
```

### Output
```bash
Task 'frozen_lake_http_rollout' batch results:
  - Rollouts: 6/8 successful
  - Success rate: 75.00%
  - Average score: 0.7500 ± 0.4330
  - Trajectory data saved to: client/evaluation_logs/trajectory_frozen_lake_http_rollout_20250610_143052.jsonl
```


## Architecture

```
┌─────────────────┐    HTTP     ┌──────────────────┐
│ Client Side     │ ◄─────────► │ Server Side      │
│ (eval-protocol)    │  Rollout    │ (Game Env)       │
│                 │             │                  │
│ • Agent Eval    │             │ • Game Logic     │
│ • Reward Func   │             │ • State Mgmt     │
│ • Trajectory    │             │ • HTTP API       │
└─────────────────┘             └──────────────────┘
```

## Project Structure

```
frozen_lake/
├── README.md                    # This overview
├── server/                      # Game Environment (HTTP API)
│   ├── README.md               # Server documentation
│   └── http_rollout_server.py  # FastAPI game server
└── client/                     # Agent Evaluation
    ├── task_def.yaml           # Task configuration (works with any model)
    ├── reward.py               # Reward function
    └── evaluation_logs/        # Generated results & trajectories
        ├── trajectory_*.jsonl  # Conversation histories + tool calls
        └── *_reeval_*.jsonl    # Re-evaluation results
```

## Game Rules

**Objective:** Navigate from S to G without falling into holes (H)

```
[S] F  F  F
 F  H  F  H
 F  F  F  H
 H  F  F  G
```

**Actions:** `"left"`, `"right"`, `"up"`, `"down"`

## Trajectory Data Format

Each trajectory JSONL file contains:

```json
{"type": "summary", "task_id": "frozen_lake_http_rollout", "num_rollouts": 8, "success_rate": 0.75, "avg_score": 0.75}
{"type": "individual_result", "rollout_index": 0, "score": 1.0, "conversation_messages": [...], "reward_function_inputs": {...}}
```

### Conversation Messages
Complete OpenAI format conversation history:
- User prompts
- Assistant responses with reasoning
- Tool calls (game actions)
- Tool results (game observations)

### Reward Function Inputs
Exact parameters passed to reward functions:
- `messages`: Full conversation history
- `state`: Game state and successful function calls
- `task_achieved`: Success/failure status
- `ground_truth`: Reference data (if available)

## Customization

### Custom Reward Functions
Create new reward functions and test them on existing trajectories:

```python
# my_rewards.py
from eval_protocol.typed_interface import reward_function
from eval_protocol.models import EvaluateResult, MetricResult

@reward_function
def efficiency_reward(messages, state=None, **kwargs):
    # Count steps taken
    step_count = len(state.get("successful_func_calls", [[]])[0])

    # Reward fewer steps
    efficiency_score = max(0.0, 1.0 - (step_count - 4) * 0.1)

    return EvaluateResult(
        score=efficiency_score,
        reason=f"Efficiency reward: {step_count} steps",
        metrics={"efficiency": MetricResult(score=efficiency_score, reason="Step efficiency")}
    )
```


## Model Performance

| Model | Success Rate | Average Score | Best Strategy |
|-------|-------------|---------------|---------------|
| qwen3-235b-a22b | 75-100% | 0.75-1.0 | down→down→right→right→down→right |
| gpt-4o-mini | 0-25% | 0.0-0.25 | Often fails at holes |

## Troubleshooting

- **Connection errors**: Server auto-starts, check port conflicts
- **API key issues**: Verify MODEL_AGENT and API key are set
- **Empty trajectories**: Check `client/evaluation_logs/` directory
- **Re-evaluation errors**: Ensure reward function module path is correct

## Next Steps

1. **Run the example**: Start with single rollout, then batch evaluation
2. **Analyze trajectories**: Examine generated JSONL files
3. **Create custom rewards**: Implement your own scoring functions
4. **Compare approaches**: Use re-evaluation to test different strategies
