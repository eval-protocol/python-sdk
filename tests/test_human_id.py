import re
import pytest

from eval_protocol.human_id import generate_id, num_combinations


def test_generate_id_basic_format():
    """Test that generate_id produces the expected adjective-noun-NN format"""
    id_str = generate_id(index=0)
    # Should match pattern: adjective-noun-NN where NN is 00-99
    assert re.match(r"^[a-z]+-[a-z]+-\d{2}$", id_str)

    # Test a few specific indices to ensure deterministic behavior
    assert generate_id(index=0) == "other-time-00"
    assert generate_id(index=1) == "other-time-01"
    assert generate_id(index=99) == "other-time-99"
    assert generate_id(index=100) == "other-year-00"


def test_generate_id_index_mapping():
    """Test that index mapping works correctly"""
    # Test number cycling (0-99)
    for i in range(100):
        id_str = generate_id(index=i)
        expected_num = f"{i:02d}"
        assert id_str.endswith(f"-{expected_num}")
        assert id_str.startswith("other-time-")

    # Test noun advancement after 100 numbers
    id_100 = generate_id(index=100)
    assert id_100.startswith("other-year-00")

    # Test adjective advancement (after all nouns * 100)
    # This will depend on dictionary size, so let's test the pattern
    from eval_protocol.human_id import dictionary

    nouns_count = len(dictionary.nouns)
    adjective_boundary = nouns_count * 100

    id_at_boundary = generate_id(index=adjective_boundary)
    # Should have advanced to the next adjective
    assert not id_at_boundary.startswith("other-")


def test_generate_id_index_out_of_range():
    """Test that invalid indices raise appropriate errors"""
    total = num_combinations()
    assert total > 0

    # Last valid index should work
    generate_id(index=total - 1)

    # First invalid index should raise error
    with pytest.raises(ValueError):
        generate_id(index=total)

    # Negative index should raise error
    with pytest.raises(ValueError):
        generate_id(index=-1)


def test_generate_id_seed_stability():
    """Test that same seed produces same ID"""
    a = generate_id(seed=1234)
    b = generate_id(seed=1234)
    assert a == b

    # Without index, default produces separator '-' and at least 3 components
    c = generate_id()

    assert re.match(r"^[a-z]+-[a-z]+-\d{2}$", c)


def test_generate_id_seed_with_index():
    """Test that seed affects index-based generation deterministically"""
    x = generate_id(index=42, seed=1)
    y = generate_id(index=42, seed=999)
    z = generate_id(index=42, seed=1)

    # Same seed should produce same result
    assert x == z
    # Different seeds should produce different results
    assert x != y

    # All should follow the correct format
    assert re.match(r"^[a-z]+-[a-z]+-\d{2}$", x)
    assert re.match(r"^[a-z]+-[a-z]+-\d{2}$", y)


def test_generate_id_random_format():
    """Test that random generation (no index) produces correct format"""
    for _ in range(10):
        id_str = generate_id()
        assert re.match(r"^[a-z]+-[a-z]+-\d{2}$", id_str)
