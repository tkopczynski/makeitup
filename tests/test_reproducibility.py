"""Tests for reproducibility using seed control."""

import pandas as pd
import pytest

from data_generation.core.generator import (
    generate_data_with_seed,
    generate_reproducibility_code,
)
from data_generation.core.quality import QualityConfig


def test_same_seed_produces_identical_data():
    """Test that using the same seed produces identical data."""
    schema = [
        {"name": "id", "type": "int", "config": {"min": 1, "max": 1000}},
        {"name": "name", "type": "name"},
        {"name": "email", "type": "email"},
        {"name": "amount", "type": "currency", "config": {"min": 10.0, "max": 500.0}},
    ]

    seed = 123456
    num_rows = 50

    # Generate data twice with same seed
    data1, seed1 = generate_data_with_seed(schema, num_rows, seed)
    data2, seed2 = generate_data_with_seed(schema, num_rows, seed)

    # Convert to DataFrames for comparison
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Verify seeds match
    assert seed1 == seed
    assert seed2 == seed

    # Verify data is identical
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    """Test that different seeds produce different data."""
    schema = [
        {"name": "id", "type": "int", "config": {"min": 1, "max": 1000}},
        {"name": "value", "type": "float", "config": {"min": 0.0, "max": 100.0}},
    ]

    num_rows = 50

    # Generate with different seeds
    data1, seed1 = generate_data_with_seed(schema, num_rows, 111111)
    data2, seed2 = generate_data_with_seed(schema, num_rows, 999999)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Verify seeds are different
    assert seed1 != seed2

    # Verify data is different (at least one column differs)
    try:
        pd.testing.assert_frame_equal(df1, df2)
        pytest.fail("DataFrames should be different with different seeds")
    except AssertionError:
        # Expected - data should be different
        pass


def test_none_seed_generates_reproducibility_code():
    """Test that None seed auto-generates a reproducibility code."""
    schema = [{"name": "id", "type": "int", "config": {"min": 1, "max": 100}}]

    data, seed = generate_data_with_seed(schema, 10, None)

    # Verify a seed was generated
    assert seed is not None
    assert isinstance(seed, int)
    assert 100000 <= seed <= 999999  # 6-digit code


def test_seed_returned_in_tuple():
    """Test that generate_data returns (data, seed) tuple."""
    schema = [{"name": "value", "type": "int"}]

    result = generate_data_with_seed(schema, 5, 555555)

    # Verify tuple structure
    assert isinstance(result, tuple)
    assert len(result) == 2
    data, seed = result
    assert isinstance(data, list)
    assert seed == 555555


def test_seed_with_quality_degradation():
    """Test that seed produces reproducible quality degradation."""
    schema = [
        {
            "name": "email",
            "type": "email",
            "config": {
                "quality_config": {
                    "null_rate": 0.2,
                    "duplicate_rate": 0.1,
                    "similar_rate": 0.1,
                    "invalid_format_rate": 0.05,
                }
            },
        },
        {
            "name": "age",
            "type": "int",
            "config": {
                "min": 18,
                "max": 80,
                "quality_config": {"null_rate": 0.1, "outlier_rate": 0.05},
            },
        },
    ]

    seed = 777777
    num_rows = 100

    # Generate twice with same seed
    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Quality issues should be identical
    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_target_generation():
    """Test that seed produces reproducible target generation."""
    schema = [
        {"name": "amount", "type": "currency", "config": {"min": 10.0, "max": 10000.0}},
        {"name": "hour", "type": "int", "config": {"min": 0, "max": 23}},
        {
            "name": "is_fraud",
            "type": "bool",
            "config": {
                "target_config": {
                    "generation_mode": "rule_based",
                    "rules": [
                        {
                            "conditions": [
                                {"feature": "amount", "operator": ">", "value": 5000},
                                {"feature": "hour", "operator": ">=", "value": 22},
                            ],
                            "probability": 0.8,
                        }
                    ],
                    "default_probability": 0.05,
                }
            },
        },
    ]

    seed = 888888
    num_rows = 50

    # Generate twice with same seed
    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Target values should be identical
    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_categories():
    """Test that seed produces reproducible category selection."""
    schema = [
        {
            "name": "status",
            "type": "category",
            "config": {"categories": ["pending", "active", "completed", "cancelled"]},
        },
        {
            "name": "priority",
            "type": "category",
            "config": {"categories": ["low", "medium", "high"]},
        },
    ]

    seed = 333333
    num_rows = 100

    # Generate twice with same seed
    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Category selections should be identical
    pd.testing.assert_frame_equal(df1, df2)


def test_seed_range_limits():
    """Test that generated seeds are within expected range."""
    for _ in range(10):
        code = generate_reproducibility_code()
        assert 100000 <= code <= 999999
        assert len(str(code)) == 6


def test_backward_compatibility_no_seed():
    """Test that not providing seed still works (backward compatibility)."""
    schema = [
        {"name": "id", "type": "int", "config": {"min": 1, "max": 100}},
        {"name": "name", "type": "name"},
    ]

    # Call without seed parameter
    data, seed = generate_data_with_seed(schema, 10)

    # Should work and generate a seed
    assert len(data) == 10
    assert seed is not None
    assert isinstance(seed, int)


def test_seed_zero():
    """Test that seed=0 works correctly."""
    schema = [{"name": "value", "type": "int", "config": {"min": 1, "max": 100}}]

    data1, seed1 = generate_data_with_seed(schema, 20, 0)
    data2, seed2 = generate_data_with_seed(schema, 20, 0)

    # Seed 0 should produce identical data
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    pd.testing.assert_frame_equal(df1, df2)


def test_very_large_seed():
    """Test that very large seeds work correctly."""
    schema = [{"name": "value", "type": "int"}]

    large_seed = 999999
    data1, seed1 = generate_data_with_seed(schema, 10, large_seed)
    data2, seed2 = generate_data_with_seed(schema, 10, large_seed)

    assert seed1 == large_seed
    assert seed2 == large_seed

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_bool_type():
    """Test that seed produces reproducible boolean generation."""
    schema = [{"name": "is_active", "type": "bool"}]

    seed = 444444
    num_rows = 100

    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_date_types():
    """Test that seed produces reproducible date generation."""
    schema = [
        {
            "name": "birth_date",
            "type": "date",
            "config": {"start_date": "1970-01-01", "end_date": "2000-12-31"},
        },
        {
            "name": "created_at",
            "type": "datetime",
            "config": {"start_date": "2020-01-01", "end_date": "2024-12-31"},
        },
    ]

    seed = 666666
    num_rows = 50

    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_text_types():
    """Test that seed produces reproducible text generation (Faker)."""
    schema = [
        {"name": "company", "type": "company"},
        {"name": "product", "type": "product"},
        {"name": "address", "type": "address"},
        {"name": "phone", "type": "phone"},
    ]

    seed = 222222
    num_rows = 30

    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Faker should produce identical results with same seed
    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_uuid():
    """Test that seed produces reproducible UUID generation."""
    schema = [{"name": "user_id", "type": "uuid"}, {"name": "session_id", "type": "uuid"}]

    seed = 151515
    num_rows = 20

    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # UUIDs should be identical with same seed
    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_percentage_type():
    """Test that seed produces reproducible percentage generation."""
    schema = [
        {
            "name": "completion_rate",
            "type": "percentage",
            "config": {"min": 0.0, "max": 100.0},
        }
    ]

    seed = 505050
    num_rows = 50

    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    pd.testing.assert_frame_equal(df1, df2)


def test_seed_with_probabilistic_target():
    """Test seed reproducibility with probabilistic target generation."""
    schema = [
        {"name": "tenure_months", "type": "int", "config": {"min": 1, "max": 120}},
        {"name": "support_tickets", "type": "int", "config": {"min": 0, "max": 20}},
        {
            "name": "will_churn",
            "type": "bool",
            "config": {
                "target_config": {
                    "generation_mode": "probabilistic",
                    "base_probability": 0.25,
                    "feature_weights": {"tenure_months": -0.002, "support_tickets": 0.03},
                    "min_probability": 0.05,
                    "max_probability": 0.90,
                }
            },
        },
    ]

    seed = 121212
    num_rows = 50

    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Probabilistic targets should be identical
    pd.testing.assert_frame_equal(df1, df2)


def test_complex_schema_reproducibility():
    """Test reproducibility with a complex multi-type schema."""
    schema = [
        {"name": "id", "type": "uuid"},
        {"name": "name", "type": "name", "config": {"text_type": "full_name"}},
        {"name": "email", "type": "email"},
        {"name": "age", "type": "int", "config": {"min": 18, "max": 80}},
        {"name": "balance", "type": "currency", "config": {"min": 0.0, "max": 10000.0}},
        {"name": "is_active", "type": "bool"},
        {
            "name": "status",
            "type": "category",
            "config": {"categories": ["active", "pending", "suspended"]},
        },
        {
            "name": "created_at",
            "type": "datetime",
            "config": {"start_date": "2020-01-01", "end_date": "2024-12-31"},
        },
    ]

    seed = 987654
    num_rows = 100

    data1, _ = generate_data_with_seed(schema, num_rows, seed)
    data2, _ = generate_data_with_seed(schema, num_rows, seed)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Complex schema should still be fully reproducible
    pd.testing.assert_frame_equal(df1, df2)
