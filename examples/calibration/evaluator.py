import math
from typing import Dict, List

import numpy as np
from openai import AsyncOpenAI

# AG News classes
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
# Mapping from class name to expected token
CLASS_TOKENS = {
    "World": " World",
    "Sports": " Sports",
    "Business": " Business",
    "Sci/Tech": " Sci",
}


async def get_logprobs(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    class_tokens: Dict[str, str],
) -> Dict[str, float]:
    """
    Get the probability of each class token given the prompt.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,  # Request enough top logprobs to hopefully find our class tokens
        )

        if not response.choices:
            return {k: 0.0 for k in class_tokens}

        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        # Create a map of token -> logprob
        token_logprobs = {tl.token: tl.logprob for tl in top_logprobs}

        # Extract probabilities for our class tokens
        probs = {}
        for class_name, token in class_tokens.items():
            if token in token_logprobs:
                probs[class_name] = math.exp(token_logprobs[token])
            else:
                probs[class_name] = 0.0

        # Normalize probabilities to sum to 1 (among the classes we care about)
        total_prob = sum(probs.values())
        if total_prob > 0:
            for k in probs:
                probs[k] /= total_prob
        else:
            # If none of the class tokens are in top_logprobs, this is a failure case.
            # We assign uniform probability or 0.
            for k in probs:
                probs[k] = 1.0 / len(class_tokens)

        return probs

    except Exception as e:
        print(f"Error calling model {model}: {e}")
        return {k: 0.0 for k in class_tokens}


def calculate_ece(preds: List[float], confs: List[float], labels: List[int], n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    preds_arr = np.array(preds)
    confs_arr = np.array(confs)
    labels_arr = np.array(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Indices of samples in this bin
        in_bin = (confs_arr > bin_lower) & (confs_arr <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(preds_arr[in_bin] == labels_arr[in_bin])
            avg_conf_in_bin = np.mean(confs_arr[in_bin])
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
