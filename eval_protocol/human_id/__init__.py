import itertools
import random
from typing import Hashable

from . import dictionary

__all__ = ["generate_id", "num_combinations"]

system_random = random.SystemRandom()


def generate_id(
    separator: str = "-",
    seed: int | float | str | bytes | bytearray | None = None,
    index: int | None = None,
) -> str:
    """
    Generate a human readable ID in format: adjective-noun-NN

    :param separator: The string to use to separate words
    :param seed: The seed to use. The same seed will produce the same ID or index-based mapping
    :param index: Optional non-negative integer providing a 1:1 mapping to an ID.
                  When provided, the mapping is deterministic and bijective for
                  all integers in range [0, total_combinations).
    :return: A human readable ID
    """

    # If a specific index is provided, use it for deterministic generation
    if index is not None:
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer if provided")

        # Prepare category lists; if seed is provided, shuffle deterministically
        if seed is not None:
            rnd = random.Random(seed)
            adjectives = tuple(rnd.sample(dictionary.adjectives, len(dictionary.adjectives)))
            nouns = tuple(rnd.sample(dictionary.nouns, len(dictionary.nouns)))
        else:
            adjectives = dictionary.adjectives
            nouns = dictionary.nouns

        # Calculate total combinations: adjectives * nouns * 100 (for 00-99)
        total = len(adjectives) * len(nouns) * 100

        if index >= total:
            raise ValueError(f"index out of range. Received {index}, max allowed is {total - 1}")

        # Decompose index into adjective, noun, and number
        number = index % 100
        remaining = index // 100
        noun_idx = remaining % len(nouns)
        adj_idx = remaining // len(nouns)

        adjective = adjectives[adj_idx]
        noun = nouns[noun_idx]

        return f"{adjective}{separator}{noun}{separator}{number:02d}"

    # Random generation
    random_obj = system_random
    if seed is not None:
        random_obj = random.Random(seed)

    adjective = random_obj.choice(dictionary.adjectives)
    noun = random_obj.choice(dictionary.nouns)
    number = random_obj.randint(0, 99)

    return f"{adjective}{separator}{noun}{separator}{number:02d}"


def num_combinations() -> int:
    """
    Return the total number of unique IDs possible.

    Format uses adjective-noun-NN, so total = adjectives * nouns * 100.
    """
    return len(dictionary.adjectives) * len(dictionary.nouns) * 100
