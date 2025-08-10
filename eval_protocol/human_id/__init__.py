import itertools
import random
from typing import Hashable

from . import dictionary

__all__ = ["generate_id"]

system_random = random.SystemRandom()


def generate_id(separator="-", seed: int | float | str | bytes | bytearray | None = None, word_count=5) -> str:
    """
    Generate a human readable ID

    :param separator: The string to use to separate words
    :param seed: The seed to use. The same seed will produce the same ID
    :param word_count: The number of words to use. Minimum of 3.
    :return: A human readable ID
    """
    if word_count < 3:
        raise ValueError("word_count cannot be lower than 3")

    random_obj = system_random
    if seed:
        random_obj = random.Random(seed)

    parts = {dictionary.verbs: 1, dictionary.adjectives: 1, dictionary.nouns: 1}

    for _ in range(3, word_count):
        parts[random_obj.choice(list(parts.keys()))] += 1

    parts = itertools.chain.from_iterable(random_obj.sample(part, count) for part, count in parts.items())

    return separator.join(parts)
