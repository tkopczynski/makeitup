"""Statistical validation tests for generated data.

These tests verify that generated data matches configured specifications
and has expected statistical properties.
"""

import re
from datetime import datetime
from pathlib import Path

import pytest

from data_generation.core.generator import generate_data


class TestNumericRangeValidation:
    """Test that numeric fields respect configured min/max bounds."""

    def test_int_range_validation(self):
        """Test all int values fall within configured range."""
        schema = [{"name": "age", "type": "int", "config": {"min": 18, "max": 65}}]
        data = generate_data(schema, 1000)

        values = [row["age"] for row in data]
        assert all(18 <= v <= 65 for v in values), "Some int values outside range"
        assert min(values) >= 18, "Min value below configured minimum"
        assert max(values) <= 65, "Max value above configured maximum"

    def test_float_range_validation(self):
        """Test all float values fall within configured range."""
        schema = [{"name": "score", "type": "float", "config": {"min": 0.0, "max": 100.0}}]
        data = generate_data(schema, 1000)

        values = [row["score"] for row in data]
        assert all(0.0 <= v <= 100.0 for v in values), "Some float values outside range"

    def test_currency_range_validation(self):
        """Test all currency values fall within configured range."""
        schema = [{"name": "price", "type": "currency", "config": {"min": 10.0, "max": 500.0}}]
        data = generate_data(schema, 1000)

        values = [row["price"] for row in data]
        assert all(10.0 <= v <= 500.0 for v in values), "Some currency values outside range"
        # Verify currency has exactly 2 decimal places
        assert all(round(v, 2) == v for v in values), "Currency values not rounded to 2 decimals"

    def test_percentage_range_validation(self):
        """Test all percentage values fall within configured range."""
        schema = [{"name": "discount", "type": "percentage"}]
        data = generate_data(schema, 1000)

        values = [row["discount"] for row in data]
        assert all(0.0 <= v <= 100.0 for v in values), "Some percentages outside 0-100 range"

    def test_percentage_custom_range(self):
        """Test percentage with custom range."""
        schema = [
            {"name": "growth", "type": "percentage", "config": {"min": -20.0, "max": 50.0}}
        ]
        data = generate_data(schema, 500)

        values = [row["growth"] for row in data]
        assert all(-20.0 <= v <= 50.0 for v in values), "Some percentages outside custom range"


class TestDistributionProperties:
    """Test that generated data has reasonable statistical distributions."""

    def test_int_distribution_coverage(self):
        """Test int values cover the configured range reasonably."""
        schema = [{"name": "value", "type": "int", "config": {"min": 1, "max": 100}}]
        data = generate_data(schema, 1000)

        values = [row["value"] for row in data]
        unique_values = set(values)

        # Should have good coverage of the range (at least 50 unique values out of 100)
        assert len(unique_values) >= 50, f"Only {len(unique_values)} unique values in range 1-100"

        # Mean should be roughly in the middle
        mean = sum(values) / len(values)
        assert 30 <= mean <= 70, f"Mean {mean} not near expected center ~50"

    def test_float_distribution_coverage(self):
        """Test float values are well distributed."""
        schema = [{"name": "value", "type": "float", "config": {"min": 0.0, "max": 1.0}}]
        data = generate_data(schema, 1000)

        values = [row["value"] for row in data]

        # Mean should be roughly in the middle
        mean = sum(values) / len(values)
        assert 0.3 <= mean <= 0.7, f"Mean {mean} not near expected center ~0.5"

        # Standard deviation should indicate spread (not all same value)
        import statistics

        std = statistics.stdev(values)
        assert std > 0.1, f"Standard deviation {std} too low, values may be clustered"

    def test_category_distribution(self):
        """Test category values are distributed across all options."""
        categories = ["A", "B", "C", "D", "E"]
        schema = [{"name": "category", "type": "category", "config": {"categories": categories}}]
        data = generate_data(schema, 1000)

        values = [row["category"] for row in data]
        unique_values = set(values)

        # All categories should appear at least once in 1000 rows
        assert unique_values == set(categories), "Not all categories appeared in generated data"

        # Each category should appear at least a few times (>10)
        from collections import Counter

        counts = Counter(values)
        assert all(
            count >= 10 for count in counts.values()
        ), f"Some categories too rare: {counts}"

    def test_bool_distribution(self):
        """Test bool values have both True and False."""
        schema = [{"name": "active", "type": "bool"}]
        data = generate_data(schema, 1000)

        values = [row["active"] for row in data]

        # Should have both True and False
        assert True in values, "No True values generated"
        assert False in values, "No False values generated"

        # Should be reasonably balanced (between 30% and 70% True)
        true_ratio = sum(values) / len(values)
        assert 0.3 <= true_ratio <= 0.7, f"Bool distribution very unbalanced: {true_ratio}"


class TestCategoryValidation:
    """Test that category fields only produce allowed values."""

    def test_category_membership(self):
        """Test all category values come from allowed list."""
        categories = ["pending", "approved", "rejected"]
        schema = [{"name": "status", "type": "category", "config": {"categories": categories}}]
        data = generate_data(schema, 500)

        values = [row["status"] for row in data]
        assert all(
            v in categories for v in values
        ), "Some category values not in allowed list"

    def test_single_category(self):
        """Test category with only one option."""
        schema = [{"name": "type", "type": "category", "config": {"categories": ["fixed"]}}]
        data = generate_data(schema, 100)

        values = [row["type"] for row in data]
        assert all(v == "fixed" for v in values), "Single category not working"

    def test_large_category_set(self):
        """Test category with many options."""
        categories = [f"cat_{i}" for i in range(100)]
        schema = [{"name": "category", "type": "category", "config": {"categories": categories}}]
        data = generate_data(schema, 1000)

        values = [row["category"] for row in data]
        assert all(v in categories for v in values), "Some values not in large category list"


class TestDateTimeValidation:
    """Test that date and datetime fields respect configured ranges."""

    def test_date_range_validation(self):
        """Test all dates fall within configured range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        schema = [
            {"name": "date", "type": "date", "config": {"start_date": start, "end_date": end}}
        ]
        data = generate_data(schema, 500)

        dates = [row["date"] for row in data]
        assert all(
            start.date() <= d <= end.date() for d in dates
        ), "Some dates outside configured range"

    def test_datetime_range_validation(self):
        """Test all datetimes fall within configured range."""
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 30)
        schema = [
            {
                "name": "timestamp",
                "type": "datetime",
                "config": {"start_date": start, "end_date": end},
            }
        ]
        data = generate_data(schema, 500)

        timestamps = [row["timestamp"] for row in data]
        assert all(
            start <= dt <= end for dt in timestamps
        ), "Some datetimes outside configured range"

    def test_date_default_range(self):
        """Test date generation with default range (last year)."""
        schema = [{"name": "created", "type": "date"}]
        data = generate_data(schema, 100)

        dates = [row["created"] for row in data]
        # All dates should be in the past year
        now = datetime.now().date()
        assert all(d <= now for d in dates), "Some dates in the future with default range"

    def test_datetime_valid_components(self):
        """Test datetime has valid components."""
        schema = [{"name": "timestamp", "type": "datetime"}]
        data = generate_data(schema, 100)

        for row in data:
            dt = row["timestamp"]
            assert 1 <= dt.month <= 12, f"Invalid month: {dt.month}"
            assert 1 <= dt.day <= 31, f"Invalid day: {dt.day}"
            assert 0 <= dt.hour <= 23, f"Invalid hour: {dt.hour}"
            assert 0 <= dt.minute <= 59, f"Invalid minute: {dt.minute}"
            assert 0 <= dt.second <= 59, f"Invalid second: {dt.second}"


class TestFormatValidation:
    """Test that formatted fields match expected patterns."""

    def test_uuid_format(self):
        """Test UUID format matches 8-4-4-4-12 pattern."""
        schema = [{"name": "id", "type": "uuid"}]
        data = generate_data(schema, 100)

        uuid_pattern = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")

        for row in data:
            assert uuid_pattern.match(
                row["id"]
            ), f"UUID {row['id']} doesn't match expected format"

    def test_email_format(self):
        """Test email format contains @ and domain."""
        schema = [{"name": "email", "type": "email"}]
        data = generate_data(schema, 100)

        for row in data:
            email = row["email"]
            assert "@" in email, f"Email {email} missing @"
            assert "." in email.split("@")[1], f"Email {email} missing domain extension"

    def test_phone_non_empty(self):
        """Test phone numbers are non-empty strings."""
        schema = [{"name": "phone", "type": "phone"}]
        data = generate_data(schema, 100)

        for row in data:
            assert isinstance(row["phone"], str), "Phone is not a string"
            assert len(row["phone"]) > 0, "Phone is empty"

    def test_text_max_length(self):
        """Test text respects max length (200 chars default)."""
        schema = [{"name": "description", "type": "text"}]
        data = generate_data(schema, 100)

        for row in data:
            assert len(row["description"]) <= 200, f"Text exceeds 200 chars: {len(row['description'])}"
            assert len(row["description"]) > 0, "Text is empty"


class TestUniquenessValidation:
    """Test that fields requiring uniqueness have no duplicates."""

    def test_uuid_uniqueness(self):
        """Test UUID fields have no duplicates."""
        schema = [{"name": "id", "type": "uuid"}]
        data = generate_data(schema, 1000)

        ids = [row["id"] for row in data]
        unique_ids = set(ids)

        assert len(ids) == len(unique_ids), f"Found {len(ids) - len(unique_ids)} duplicate UUIDs"

    def test_multiple_uuid_fields(self):
        """Test multiple UUID fields are all unique."""
        schema = [{"name": "id1", "type": "uuid"}, {"name": "id2", "type": "uuid"}]
        data = generate_data(schema, 500)

        # Each field should have unique values
        id1s = [row["id1"] for row in data]
        id2s = [row["id2"] for row in data]

        assert len(id1s) == len(set(id1s)), "Duplicate UUIDs in id1"
        assert len(id2s) == len(set(id2s)), "Duplicate UUIDs in id2"

        # Fields should be different from each other
        assert set(id1s).isdisjoint(set(id2s)), "UUID fields have overlapping values"


class TestReferenceIntegrity:
    """Test that reference fields maintain integrity with parent tables."""

    def test_reference_values_exist_in_parent(self, tmp_path):
        """Test all reference values exist in parent table."""
        # Create parent CSV
        parent_file = tmp_path / "users.csv"
        parent_file.write_text("user_id,name\n1,Alice\n2,Bob\n3,Charlie\n")

        schema = [
            {"name": "transaction_id", "type": "uuid"},
            {
                "name": "user_id",
                "type": "reference",
                "config": {"reference_file": str(parent_file), "reference_column": "user_id"},
            },
        ]
        data = generate_data(schema, 200)

        valid_user_ids = {"1", "2", "3"}  # CSV reads as strings
        user_ids = [str(row["user_id"]) for row in data]

        assert all(
            uid in valid_user_ids for uid in user_ids
        ), "Some reference values not in parent table"

    def test_reference_distribution(self, tmp_path):
        """Test reference values are distributed across parent values."""
        # Create parent CSV with 10 values
        parent_file = tmp_path / "products.csv"
        product_ids = "\n".join([f"{i},Product{i}" for i in range(1, 11)])
        parent_file.write_text(f"product_id,name\n{product_ids}\n")

        schema = [
            {
                "name": "product_id",
                "type": "reference",
                "config": {"reference_file": str(parent_file), "reference_column": "product_id"},
            }
        ]
        data = generate_data(schema, 500)

        product_ids_used = set(str(row["product_id"]) for row in data)

        # Should use at least 70% of available parent values
        assert (
            len(product_ids_used) >= 7
        ), f"Only {len(product_ids_used)} of 10 parent values used"

    def test_multiple_references_to_same_table(self, tmp_path):
        """Test multiple reference fields to the same parent table."""
        parent_file = tmp_path / "categories.csv"
        parent_file.write_text("cat_id,name\nA,Alpha\nB,Beta\nC,Gamma\n")

        schema = [
            {
                "name": "primary_cat",
                "type": "reference",
                "config": {"reference_file": str(parent_file), "reference_column": "cat_id"},
            },
            {
                "name": "secondary_cat",
                "type": "reference",
                "config": {"reference_file": str(parent_file), "reference_column": "cat_id"},
            },
        ]
        data = generate_data(schema, 200)

        valid_cats = {"A", "B", "C"}
        primary_cats = [str(row["primary_cat"]) for row in data]
        secondary_cats = [str(row["secondary_cat"]) for row in data]

        assert all(c in valid_cats for c in primary_cats), "Invalid primary category"
        assert all(c in valid_cats for c in secondary_cats), "Invalid secondary category"


class TestComplexSchemas:
    """Test statistical properties with complex multi-field schemas."""

    def test_all_field_types_together(self):
        """Test schema with all field types generates valid data."""
        schema = [
            {"name": "id", "type": "uuid"},
            {"name": "name", "type": "name"},
            {"name": "email", "type": "email"},
            {"name": "age", "type": "int", "config": {"min": 18, "max": 80}},
            {"name": "balance", "type": "currency", "config": {"min": 0.0, "max": 10000.0}},
            {
                "name": "status",
                "type": "category",
                "config": {"categories": ["active", "inactive", "pending"]},
            },
            {"name": "is_verified", "type": "bool"},
            {"name": "created_at", "type": "datetime"},
            {"name": "score", "type": "percentage"},
        ]
        data = generate_data(schema, 500)

        assert len(data) == 500
        # Verify each row has all fields
        for row in data:
            assert len(row) == 9, f"Row missing fields: {row.keys()}"

    def test_large_dataset_performance(self):
        """Test generating large dataset completes in reasonable time."""
        import time

        schema = [
            {"name": "id", "type": "uuid"},
            {"name": "value1", "type": "int"},
            {"name": "value2", "type": "float"},
            {"name": "category", "type": "category", "config": {"categories": ["A", "B", "C"]}},
        ]

        start = time.time()
        data = generate_data(schema, 10000)
        elapsed = time.time() - start

        assert len(data) == 10000
        # Should complete in under 10 seconds
        assert elapsed < 10, f"Generation took {elapsed:.2f}s, expected < 10s"
