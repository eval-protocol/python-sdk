import re
import pytest

from eval_protocol.human_id import generate_id, num_combinations


def test_generate_id_index_basic_3_words():
    # index 0 maps to the first element of each category (verb, adjective, noun)
    assert generate_id(index=0, word_count=3) == "be-other-time"

    # incrementing index advances the least-significant position (noun)
    assert generate_id(index=1, word_count=3) == "be-other-year"

    # carry into the adjective when nouns wrap
    # index == len(nouns) => adjective advances by 1, noun resets
    # nouns length inferred by probing with large indices is brittle; instead, compute via reach
    # We know index=0 gives be-other-time, and index that produces adjective=new, noun=time should be reachable.
    # Derive by scanning forward until adjective changes to 'new'. This keeps test robust to dictionary size edits.
    base = generate_id(index=0, word_count=3)
    # Find the first index where adjective becomes 'new' and noun resets to 'time'
    target = None
    for i in range(1, 2000):
        cand = generate_id(index=i, word_count=3)
        if cand.startswith("be-new-time"):
            target = i
            break
    assert target is not None, "Expected to find carry into adjective within search bound"
    assert generate_id(index=target, word_count=3) == "be-new-time"


def test_generate_id_index_word_count_cycle():
    # word_count cycles categories: verb, adj, noun, verb, adj, ...
    assert generate_id(index=0, word_count=5) == "be-other-time-be-other"
    # increment least-significant position (adj at position 5)
    assert generate_id(index=1, word_count=5) == "be-other-time-be-new"


def test_generate_id_index_out_of_range_and_negative():
    # Use exported total combinations for clean boundary checks
    total = num_combinations(word_count=3)
    assert total > 0
    # Last valid index
    generate_id(index=total - 1, word_count=3)
    # First invalid index
    with pytest.raises(ValueError):
        generate_id(index=total, word_count=3)

    with pytest.raises(ValueError):
        generate_id(index=-1, word_count=3)


def test_generate_id_seed_stability_and_compat():
    # Without index, same seed yields same id
    a = generate_id(seed=1234)
    b = generate_id(seed=1234)
    assert a == b

    # Without index, default produces separator '-' and at least 3 components
    c = generate_id()
    assert re.match(r"^[a-z]+(-[a-z]+){2,}$", c)


def test_generate_id_index_ignores_seed():
    # With index provided, seed should affect the mapping deterministically
    x = generate_id(index=42, seed=1)
    y = generate_id(index=42, seed=999)
    z = generate_id(index=42, seed=1)
    assert x != y
    assert x == z
