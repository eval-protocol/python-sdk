"""
Debugging script for repetition rewards.
"""

from eval_protocol.rewards.repetition import repetition_penalty_reward


def main():
    """Run a basic test of repetition_penalty_reward with an empty response."""
    messages = [
        {"role": "user", "content": "Write a response"},
        {"role": "assistant", "content": ""},
    ]

    result = repetition_penalty_reward(messages=messages)
    print(f"Result score: {result['score']}")
    print(f"Result structure: {result}")


if __name__ == "__main__":
    main()
