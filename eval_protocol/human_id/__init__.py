import itertools
import random
from typing import Hashable

from . import dictionary

__all__ = ["generate_id", "num_combinations"]

system_random = random.SystemRandom()


def generate_id(
    separator: str = "-",
    seed: int | float | str | bytes | bytearray | None = None,
    word_count: int = 5,
    index: int | None = None,
) -> str:
    """
    Generate a human readable ID

    :param separator: The string to use to separate words
    :param seed: The seed to use. The same seed will produce the same ID or index-based mapping
    :param index: Optional non-negative integer providing a 1:1 mapping to an ID.
                  When provided, the mapping is deterministic and bijective for
                  all integers in range [0, total_combinations).
    :param word_count: The number of words to use. Minimum of 3.
    :return: A human readable ID
    """
    if word_count < 3:
        raise ValueError("word_count cannot be lower than 3")

    # If a specific index is provided, use mixed-radix encoding into a fixed
    # sequence of parts to guarantee a bijection between integers and IDs.
    # The sequence cycles as: verb, adjective, noun, verb, adjective, noun, ...
    if index is not None:
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer if provided")

        # Prepare category lists; if seed is provided, shuffle deterministically
        base_categories = [dictionary.verbs, dictionary.adjectives, dictionary.nouns]
        if seed is not None:
            rnd = random.Random(seed)
            categories = [tuple(rnd.sample(cat, len(cat))) for cat in base_categories]
        else:
            categories = base_categories
        # Build the category order for the desired word_count
        ordered_categories = [categories[i % 3] for i in range(word_count)]

        # Compute total number of combinations for this word_count
        radices = [len(cat) for cat in ordered_categories]
        total = num_combinations(word_count)

        if index >= total:
            raise ValueError(f"index out of range for given word_count. Received {index}, max allowed is {total - 1}")

        # Mixed-radix decomposition (least significant position is the last word)
        digits: list[int] = []
        remaining = index
        for base in reversed(radices):
            digits.append(remaining % base)
            remaining //= base
        digits.reverse()

        words = [ordered_categories[pos][digits[pos]] for pos in range(word_count)]
        return separator.join(words)

    random_obj = system_random
    if seed is not None:
        random_obj = random.Random(seed)

    parts = {dictionary.verbs: 1, dictionary.adjectives: 1, dictionary.nouns: 1}

    for _ in range(3, word_count):
        parts[random_obj.choice(list(parts.keys()))] += 1

    parts = itertools.chain.from_iterable(random_obj.sample(part, count) for part, count in parts.items())

    return separator.join(parts)


def num_combinations(word_count: int = 5) -> int:
    """
    Return the total number of unique IDs possible for the given word_count.

    The sequence of categories cycles as: verb, adjective, noun, then repeats.
    This value can be used to mod an index when calling generate_id(index=...).
    """
    if word_count < 3:
        raise ValueError("word_count cannot be lower than 3")

    categories = [dictionary.verbs, dictionary.adjectives, dictionary.nouns]
    radices = [len(categories[i % 3]) for i in range(word_count)]
    total = 1
    for r in radices:
        total *= r
    return total
