import os

import dspy
from dspy.clients.lm import LM

REFLECTION_LM_CONFIGS = {
    "gpt-5": {
        "model": "gpt-5",
        "temperature": 1.0,
        "max_tokens": 32000,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
    },
    "kimi-k2-instruct-0905": {
        "model": "accounts/fireworks/models/kimi-k2-instruct-0905",
        "temperature": 0.6,  # Kimi recommended temperature
        "max_tokens": 131000,
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
}


def build_reflection_lm(reflection_lm_name: str) -> LM:
    reflection_lm_config = REFLECTION_LM_CONFIGS[reflection_lm_name]
    return dspy.LM(
        model=reflection_lm_config["model"],
        temperature=reflection_lm_config["temperature"],
        max_tokens=reflection_lm_config["max_tokens"],
        api_key=reflection_lm_config["api_key"],
        base_url=reflection_lm_config["base_url"],
    )
