"""ML validation tests for generated data.

These tests verify that generated data is suitable for machine learning:
- Class balance validation
- Feature-target correlation
- Data leakage detection
- Data splitting feasibility
- Feature variance and informativeness
"""

import statistics
from collections import Counter

import pytest

from data_generation.core.generator import generate_data


class TestClassBalanceValidation:
    """Test that classification targets have appropriate class balance."""

    def test_binary_classification_balance(self):
        """Test binary classification has reasonable balance."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 1.0}},
            {"name": "feature2", "type": "int", "config": {"min": 1, "max": 100}},
            {"name": "is_positive", "type": "bool"},
        ]
        data = generate_data(schema, 1000)

        # Count positive class
        positive_count = sum(1 for row in data if row["is_positive"])
        positive_rate = positive_count / len(data)

        # Should be reasonably balanced (30-70%)
        assert (
            0.3 <= positive_rate <= 0.7
        ), f"Class balance {positive_rate} outside reasonable range"

    def test_multiclass_classification_balance(self):
        """Test multi-class classification has all classes represented."""
        categories = ["class_a", "class_b", "class_c", "class_d"]
        schema = [
            {"name": "feature1", "type": "float"},
            {
                "name": "target_class",
                "type": "category",
                "config": {"categories": categories},
            },
        ]
        data = generate_data(schema, 1000)

        # Count each class
        class_counts = Counter(row["target_class"] for row in data)

        # All classes should appear
        assert set(class_counts.keys()) == set(categories), "Not all classes present"

        # Each class should have at least 10% of data
        for cls, count in class_counts.items():
            proportion = count / len(data)
            assert proportion >= 0.1, f"Class {cls} has only {proportion:.1%} of data"

    def test_imbalanced_classification_feasible(self):
        """Test that minority class has sufficient samples."""
        # Simulate 10% minority class
        schema = [
            {"name": "feature1", "type": "float"},
            {"name": "feature2", "type": "int"},
            {"name": "is_fraud", "type": "bool"},
        ]
        data = generate_data(schema, 1000)

        # Count minority class (whichever is smaller)
        positive_count = sum(1 for row in data if row["is_fraud"])
        negative_count = len(data) - positive_count
        minority_count = min(positive_count, negative_count)

        # Minority class should have at least 50 samples for meaningful training
        assert (
            minority_count >= 50
        ), f"Minority class has only {minority_count} samples, need at least 50"

    def test_stratified_split_feasibility(self):
        """Test that stratified splitting is feasible."""
        categories = ["A", "B", "C"]
        schema = [
            {"name": "feature", "type": "float"},
            {"name": "target", "type": "category", "config": {"categories": categories}},
        ]
        data = generate_data(schema, 300)

        # Count each class
        class_counts = Counter(row["target"] for row in data)

        # Each class should have enough samples for 70/30 split
        # Minimum 10 samples in test set (30% of 300 = 90 total test samples)
        for cls, count in class_counts.items():
            expected_test_samples = count * 0.3
            assert (
                expected_test_samples >= 5
            ), f"Class {cls} would have only {expected_test_samples:.0f} test samples"


class TestFeatureCorrelation:
    """Test feature-target relationships and inter-feature correlations."""

    def test_features_have_variance(self):
        """Test that features are not constant."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "feature2", "type": "int", "config": {"min": 1, "max": 50}},
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 500)

        # Extract feature values
        feature1_vals = [row["feature1"] for row in data]
        feature2_vals = [row["feature2"] for row in data]

        # Calculate standard deviation
        std1 = statistics.stdev(feature1_vals)
        std2 = statistics.stdev(feature2_vals)

        # Features should have meaningful variance
        assert std1 > 0, "Feature1 has no variance"
        assert std2 > 0, "Feature2 has no variance"

        # For our ranges, expect reasonable spread
        assert std1 > 10, f"Feature1 std {std1} too low for range 0-100"
        assert std2 > 5, f"Feature2 std {std2} too low for range 1-50"

    def test_no_perfect_correlation(self):
        """Test that no feature perfectly predicts the target."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 1.0}},
            {"name": "feature2", "type": "int", "config": {"min": 1, "max": 100}},
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 500)

        # Check that no single feature perfectly predicts target
        # Group by target and check feature distributions overlap

        target_true = [row["feature1"] for row in data if row["target"]]
        target_false = [row["feature1"] for row in data if not row["target"]]

        # Both groups should have some data
        assert len(target_true) > 0 and len(target_false) > 0, "Target has no variance"

        # Calculate ranges
        true_range = (min(target_true), max(target_true))
        false_range = (min(target_false), max(target_false))

        # Ranges should overlap (indicating imperfect correlation)
        ranges_overlap = true_range[0] <= false_range[1] and false_range[0] <= true_range[1]
        assert ranges_overlap, "Feature1 perfectly separates classes (possible data leakage)"

    def test_numeric_features_not_identical(self):
        """Test that different numeric features are not identical."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "feature3", "type": "float", "config": {"min": 0.0, "max": 100.0}},
        ]
        data = generate_data(schema, 100)

        # Extract features
        f1 = [row["feature1"] for row in data]
        f2 = [row["feature2"] for row in data]
        f3 = [row["feature3"] for row in data]

        # Features should not be identical
        assert f1 != f2, "Feature1 and Feature2 are identical"
        assert f1 != f3, "Feature1 and Feature3 are identical"
        assert f2 != f3, "Feature2 and Feature3 are identical"

    def test_category_features_have_diversity(self):
        """Test that categorical features have multiple values."""
        schema = [
            {
                "name": "category",
                "type": "category",
                "config": {"categories": ["A", "B", "C", "D", "E"]},
            },
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 500)

        categories = [row["category"] for row in data]
        unique_categories = set(categories)

        # Should use most of the available categories
        assert (
            len(unique_categories) >= 4
        ), f"Only {len(unique_categories)} unique categories out of 5"


class TestDataLeakageDetection:
    """Test for potential data leakage issues."""

    def test_no_identical_feature_and_target(self):
        """Test that no feature is identical to target."""
        schema = [
            {"name": "feature", "type": "bool"},
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 200)

        # Extract values
        features = [row["feature"] for row in data]
        targets = [row["target"] for row in data]

        # They should not be identical
        assert features != targets, "Feature is identical to target (data leakage)"

    def test_uuid_fields_are_unique(self):
        """Test that UUID fields don't leak information."""
        schema = [
            {"name": "id", "type": "uuid"},
            {"name": "feature", "type": "float"},
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 500)

        ids = [row["id"] for row in data]

        # All IDs should be unique (no information leakage through ID patterns)
        assert len(ids) == len(set(ids)), "UUIDs are not unique"


class TestDataSplittingValidation:
    """Test that data can be properly split for ML training."""

    def test_sufficient_samples_for_split(self):
        """Test that dataset has enough samples for train/test split."""
        schema = [
            {"name": "feature1", "type": "float"},
            {"name": "feature2", "type": "int"},
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 100)

        # For ML, minimum recommended is ~100 samples
        assert len(data) >= 100, f"Only {len(data)} samples, recommend at least 100"

        # Simulate 70/30 split
        train_size = int(len(data) * 0.7)
        test_size = len(data) - train_size

        # Both splits should be reasonable
        assert train_size >= 50, f"Train set only {train_size} samples"
        assert test_size >= 20, f"Test set only {test_size} samples"

    def test_no_duplicate_rows(self):
        """Test that there are no exact duplicate rows (which could leak into test set)."""
        schema = [
            {"name": "feature1", "type": "int", "config": {"min": 1, "max": 1000}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 500)

        # Convert rows to tuples for hashing
        row_tuples = [
            tuple((k, v) for k, v in sorted(row.items()) if k != "target") for row in data
        ]

        # Should have mostly unique feature combinations
        unique_rows = len(set(row_tuples))
        duplicate_rate = 1 - (unique_rows / len(data))

        # Allow some duplicates due to randomness, but not too many
        assert (
            duplicate_rate < 0.1
        ), f"Too many duplicate rows: {duplicate_rate:.1%} (>10%)"

    def test_reference_integrity_for_splitting(self):
        """Test that reference fields maintain integrity when splitting."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create parent table
            parent_file = Path(tmpdir) / "users.csv"
            parent_file.write_text("user_id,name\n1,Alice\n2,Bob\n3,Charlie\n")

            schema = [
                {"name": "transaction_id", "type": "uuid"},
                {
                    "name": "user_id",
                    "type": "reference",
                    "config": {
                        "reference_file": str(parent_file),
                        "reference_column": "user_id",
                    },
                },
                {"name": "amount", "type": "currency"},
            ]
            data = generate_data(schema, 100)

            # All user_ids should be valid references
            user_ids = [str(row["user_id"]) for row in data]
            valid_ids = {"1", "2", "3"}

            assert all(
                uid in valid_ids for uid in user_ids
            ), "Some reference values are invalid"


class TestFeatureQuality:
    """Test feature quality for ML readiness."""

    def test_no_all_null_features(self):
        """Test that features are not all null."""
        schema = [
            {
                "name": "feature1",
                "type": "float",
                "config": {"quality_config": {"null_rate": 0.2}},
            },
            {
                "name": "feature2",
                "type": "int",
                "config": {"quality_config": {"null_rate": 0.1}},
            },
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 500)

        # Count nulls per feature
        feature1_nulls = sum(1 for row in data if row["feature1"] is None)
        feature2_nulls = sum(1 for row in data if row["feature2"] is None)

        # Should not be all null
        assert feature1_nulls < len(data), "Feature1 is all null"
        assert feature2_nulls < len(data), "Feature2 is all null"

        # Should have some non-null values for training
        assert feature1_nulls < len(data) * 0.9, "Feature1 >90% null, too sparse"
        assert feature2_nulls < len(data) * 0.9, "Feature2 >90% null, too sparse"

    def test_reasonable_missing_data_rate(self):
        """Test that overall missing data rate is manageable."""
        schema = [
            {
                "name": "feature1",
                "type": "float",
                "config": {"quality_config": {"null_rate": 0.1}},
            },
            {
                "name": "feature2",
                "type": "int",
                "config": {"quality_config": {"null_rate": 0.15}},
            },
            {
                "name": "feature3",
                "type": "text",
                "config": {"quality_config": {"null_rate": 0.05}},
            },
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 1000)

        # Calculate overall missing rate
        total_values = len(data) * 3  # 3 features with quality config
        missing_values = sum(
            1
            for row in data
            for field in ["feature1", "feature2", "feature3"]
            if row[field] is None
        )
        missing_rate = missing_values / total_values

        # Overall missing rate should be reasonable for ML (<30%)
        assert missing_rate < 0.3, f"Overall missing rate {missing_rate:.1%} too high (>30%)"

    def test_features_with_different_types(self):
        """Test that dataset has mix of feature types."""
        schema = [
            {"name": "numeric_feature", "type": "float"},
            {"name": "integer_feature", "type": "int"},
            {
                "name": "categorical_feature",
                "type": "category",
                "config": {"categories": ["A", "B", "C"]},
            },
            {"name": "boolean_feature", "type": "bool"},
            {"name": "target", "type": "bool"},
        ]
        data = generate_data(schema, 100)

        # Verify we have different data types
        sample = data[0]

        assert isinstance(sample["numeric_feature"], float), "Numeric feature not float"
        assert isinstance(sample["integer_feature"], int), "Integer feature not int"
        assert sample["categorical_feature"] in ["A", "B", "C"], "Category invalid"
        assert isinstance(sample["boolean_feature"], bool), "Boolean feature not bool"


class TestMLReadinessScenarios:
    """Test complete ML scenarios."""

    def test_binary_classification_dataset(self):
        """Test a complete binary classification dataset."""
        schema = [
            {"name": "id", "type": "uuid"},
            {"name": "age", "type": "int", "config": {"min": 18, "max": 80}},
            {"name": "income", "type": "currency", "config": {"min": 20000, "max": 200000}},
            {
                "name": "education",
                "type": "category",
                "config": {"categories": ["high_school", "bachelor", "master", "phd"]},
            },
            {"name": "is_customer", "type": "bool"},
        ]
        data = generate_data(schema, 500)

        # Validate dataset structure
        assert len(data) == 500

        # Check class balance
        positive_count = sum(1 for row in data if row["is_customer"])
        balance = positive_count / len(data)
        assert 0.2 <= balance <= 0.8, f"Class balance {balance} may be problematic"

        # Check feature ranges
        ages = [row["age"] for row in data]
        assert all(18 <= age <= 80 for age in ages), "Ages outside expected range"

        # Check education diversity
        educations = set(row["education"] for row in data)
        assert len(educations) >= 3, "Education categories too limited"

    def test_regression_dataset(self):
        """Test a complete regression dataset."""
        schema = [
            {"name": "square_feet", "type": "int", "config": {"min": 500, "max": 5000}},
            {"name": "bedrooms", "type": "int", "config": {"min": 1, "max": 5}},
            {"name": "bathrooms", "type": "float", "config": {"min": 1.0, "max": 4.0}},
            {"name": "age_years", "type": "int", "config": {"min": 0, "max": 100}},
            {"name": "price", "type": "currency", "config": {"min": 100000, "max": 1000000}},
        ]
        data = generate_data(schema, 300)

        # Validate dataset
        assert len(data) == 300

        # All features should have variance
        square_feet = [row["square_feet"] for row in data]
        prices = [row["price"] for row in data]

        assert len(set(square_feet)) > 100, "Square feet has limited diversity"
        assert len(set(prices)) > 100, "Prices have limited diversity"

        # Features should be in reasonable ranges
        assert all(500 <= sf <= 5000 for sf in square_feet), "Square feet out of range"
        assert all(100000 <= p <= 1000000 for p in prices), "Prices out of range"

    def test_multiclass_classification_dataset(self):
        """Test a complete multi-class classification dataset."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature3", "type": "int", "config": {"min": 1, "max": 100}},
            {
                "name": "class",
                "type": "category",
                "config": {"categories": ["class_0", "class_1", "class_2", "class_3"]},
            },
        ]
        data = generate_data(schema, 400)

        # Check all classes present
        classes = [row["class"] for row in data]
        class_counts = Counter(classes)

        assert len(class_counts) == 4, "Not all classes present"

        # Each class should have reasonable representation
        for cls, count in class_counts.items():
            proportion = count / len(data)
            assert proportion >= 0.15, f"Class {cls} underrepresented: {proportion:.1%}"
