import json
import matplotlib.pyplot as plt
import os

metrics_file = "/tmp/eval_protocol_integration_v9/metrics.jsonl"
output_file = "python-sdk/examples/tinker_math_rl/reward_plot_integration_v9.png"

steps = []
accuracies = []
rewards = []
formats = []

with open(metrics_file, "r") as f:
    for line in f:
        data = json.loads(line)
        # Only plot training steps (where "step" is present and usually matches training_client/step)
        # Some lines might be eval steps (which also have "step" but different keys).
        # Let's check if it's a training step or eval step.

        # Based on the log output:
        # Eval lines look like: {"step": 0, "test/env/all/reward/total": ...}
        # Train lines look like: {"step": 0, "env/all/reward/total": ...}

        if "env/all/reward/total" in data and "test/env/all/reward/total" not in data:
            steps.append(data["step"])
            rewards.append(data["env/all/reward/total"])
            accuracies.append(data.get("env/all/correct", 0.0))
            formats.append(data.get("env/all/format", 0.0))

plt.figure(figsize=(10, 6))
plt.plot(steps, accuracies, label="Accuracy", marker="o")
plt.plot(steps, rewards, label="Total Reward", marker="o")
plt.plot(steps, formats, label="Format Compliance", marker="o")

plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Training Metrics: Eval Protocol Integration V9")
plt.legend()
plt.grid(True)
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
