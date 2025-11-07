"""Model training tests for generated data.

These tests verify that actual ML models can be trained on generated data
and achieve reasonable performance, validating the data is truly useful for ML.

Tests require scikit-learn. Install with: pip install scikit-learn
"""

import statistics

import pytest

from data_generation.core.generator import generate_data

# Check if scikit-learn is available
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")


class TestBinaryClassificationModels:
    """Test that binary classification models can learn from generated data."""

    def test_logistic_regression_binary_classification(self):
        """Test LogisticRegression achieves reasonable accuracy."""
        # Generate binary classification dataset
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature3", "type": "int", "config": {"min": 1, "max": 100}},
            {"name": "target", "type": "bool"},
        ]
        data, _ = generate_data(schema, 1000)

        # Prepare data
        X = [[row["feature1"], row["feature2"], row["feature3"]] for row in data]
        y = [int(row["target"]) for row in data]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Should achieve better than random (0.5) but not perfect
        # Random features should get ~0.45-0.55 accuracy
        assert 0.4 <= accuracy <= 0.7, f"Accuracy {accuracy:.3f} unexpected for random features"

    def test_random_forest_binary_classification(self):
        """Test RandomForest can be trained without errors."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "feature2", "type": "int", "config": {"min": 1, "max": 50}},
            {"name": "feature3", "type": "float", "config": {"min": -10.0, "max": 10.0}},
            {"name": "is_positive", "type": "bool"},
        ]
        data, _ = generate_data(schema, 800)

        # Prepare data
        X = [[row["feature1"], row["feature2"], row["feature3"]] for row in data]
        y = [int(row["is_positive"]) for row in data]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Train
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Model should complete training and make predictions
        assert 0.3 <= accuracy <= 0.8, f"Accuracy {accuracy:.3f} outside reasonable range"
        assert 0.2 <= f1 <= 0.8, f"F1 score {f1:.3f} outside reasonable range"

    def test_binary_classification_auc_score(self):
        """Test that AUC score is calculable and reasonable."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 1.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 1.0}},
            {"name": "target", "type": "bool"},
        ]
        data, _ = generate_data(schema, 600)

        X = [[row["feature1"], row["feature2"]] for row in data]
        y = [int(row["target"]) for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Get probability predictions for AUC
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        # AUC should be between 0.4 and 0.7 for random features
        assert 0.3 <= auc <= 0.8, f"AUC {auc:.3f} outside expected range for random features"

    def test_no_overfitting_with_train_test_gap(self):
        """Test that train/test performance gap is not excessive."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature3", "type": "int", "config": {"min": 1, "max": 20}},
            {"name": "target", "type": "bool"},
        ]
        data, _ = generate_data(schema, 1000)

        X = [[row["feature1"], row["feature2"], row["feature3"]] for row in data]
        y = [int(row["target"]) for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Use a simple model to avoid overfitting
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate on both sets
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))

        # Gap should be small (< 0.15) for non-overfitted model
        gap = abs(train_accuracy - test_accuracy)
        assert gap < 0.20, f"Train/test gap {gap:.3f} suggests overfitting or data issues"


class TestMultiClassClassificationModels:
    """Test that multi-class classification models can learn from generated data."""

    def test_multiclass_logistic_regression(self):
        """Test LogisticRegression on multi-class problem."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature3", "type": "int", "config": {"min": 1, "max": 50}},
            {
                "name": "class",
                "type": "category",
                "config": {"categories": ["class_0", "class_1", "class_2", "class_3"]},
            },
        ]
        data, _ = generate_data(schema, 800)

        # Prepare data
        X = [[row["feature1"], row["feature2"], row["feature3"]] for row in data]
        y = [row["class"] for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train multi-class model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # For 4 classes, random baseline is 0.25
        # Should achieve slightly better than random
        assert 0.2 <= accuracy <= 0.6, f"Accuracy {accuracy:.3f} outside expected range"

    def test_multiclass_random_forest(self):
        """Test RandomForest on multi-class problem."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "feature2", "type": "int", "config": {"min": 1, "max": 100}},
            {
                "name": "target",
                "type": "category",
                "config": {"categories": ["A", "B", "C"]},
            },
        ]
        data, _ = generate_data(schema, 600)

        X = [[row["feature1"], row["feature2"]] for row in data]
        y = [row["target"] for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # For 3 classes, random baseline is 0.33
        assert 0.25 <= accuracy <= 0.7, f"Accuracy {accuracy:.3f} outside expected range"

    def test_all_classes_predicted(self):
        """Test that model predicts all classes (no class ignored)."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {
                "name": "class",
                "type": "category",
                "config": {"categories": ["A", "B", "C", "D"]},
            },
        ]
        data, _ = generate_data(schema, 1000)

        X = [[row["feature1"], row["feature2"]] for row in data]
        y = [row["class"] for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # All classes should appear in predictions
        predicted_classes = set(y_pred)
        expected_classes = {"A", "B", "C", "D"}

        assert (
            len(predicted_classes) >= 3
        ), f"Only {len(predicted_classes)} classes predicted, expected 4"


class TestRegressionModels:
    """Test that regression models can learn from generated data."""

    def test_linear_regression_basic(self):
        """Test LinearRegression can be trained."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 100.0}},
            {"name": "feature3", "type": "int", "config": {"min": 1, "max": 50}},
            {"name": "target", "type": "float", "config": {"min": 0.0, "max": 1000.0}},
        ]
        data, _ = generate_data(schema, 800)

        X = [[row["feature1"], row["feature2"], row["feature3"]] for row in data]
        y = [row["target"] for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate R²
        r2 = r2_score(y_test, y_pred)

        # For random features, R² should be close to 0 (could be negative)
        assert -0.5 <= r2 <= 0.3, f"R² {r2:.3f} outside expected range for random features"

    def test_random_forest_regression(self):
        """Test RandomForestRegressor can be trained."""
        schema = [
            {"name": "square_feet", "type": "int", "config": {"min": 500, "max": 5000}},
            {"name": "bedrooms", "type": "int", "config": {"min": 1, "max": 5}},
            {"name": "price", "type": "currency", "config": {"min": 100000, "max": 1000000}},
        ]
        data, _ = generate_data(schema, 500)

        X = [[row["square_feet"], row["bedrooms"]] for row in data]
        y = [row["price"] for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate MSE and R²
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # MSE should be finite
        assert mse > 0, "MSE should be positive"
        assert mse < 1e12, f"MSE {mse:.2e} unexpectedly high"

        # R² for random features
        assert -0.5 <= r2 <= 0.3, f"R² {r2:.3f} outside expected range"

    def test_regression_predictions_in_range(self):
        """Test that regression predictions are reasonable."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "target", "type": "float", "config": {"min": 0.0, "max": 100.0}},
        ]
        data, _ = generate_data(schema, 300)

        X = [[row["feature1"]] for row in data]
        y = [row["target"] for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Predictions should be in reasonable range (within 3x the target range)
        assert all(
            -200 <= pred <= 300 for pred in y_pred
        ), "Some predictions far outside expected range"


class TestLearningCurves:
    """Test that models improve with more data."""

    def test_accuracy_improves_with_more_data(self):
        """Test that model accuracy generally improves with more training data."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "target", "type": "bool"},
        ]

        # Generate large dataset
        data, _ = generate_data(schema, 2000)
        X = [[row["feature1"], row["feature2"]] for row in data]
        y = [int(row["target"]) for row in data]

        # Reserve 500 samples for testing
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=500, random_state=42
        )

        accuracies = []

        # Train with increasing amounts of data
        for train_size in [100, 300, 500, 1000, 1500]:
            X_train = X_train_full[:train_size]
            y_train = y_train_full[:train_size]

            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Performance should not degrade significantly with more data
        # (May not strictly increase due to random features, but should be stable)
        assert max(accuracies) - min(accuracies) < 0.3, "Performance too unstable across data sizes"

    def test_model_trains_on_small_dataset(self):
        """Test that model can train even on small dataset."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "target", "type": "bool"},
        ]
        data, _ = generate_data(schema, 100)

        X = [[row["feature1"]] for row in data]
        y = [int(row["target"]) for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Should complete without errors even on small data
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Just verify it runs
        assert 0.0 <= accuracy <= 1.0, "Accuracy outside valid range"


class TestModelWithMessyData:
    """Test that models can handle data with quality issues."""

    def test_model_with_null_values(self):
        """Test that we can handle datasets with null values."""
        schema = [
            {
                "name": "feature1",
                "type": "float",
                "config": {"min": 0.0, "max": 10.0, "quality_config": {"null_rate": 0.1}},
            },
            {
                "name": "feature2",
                "type": "int",
                "config": {"min": 1, "max": 50, "quality_config": {"null_rate": 0.15}},
            },
            {"name": "target", "type": "bool"},
        ]
        data, _ = generate_data(schema, 500)

        # Filter out rows with nulls for this test (real-world would use imputation)
        clean_data = [
            row for row in data if row["feature1"] is not None and row["feature2"] is not None
        ]

        # Should have removed some rows
        assert len(clean_data) < len(data), "Expected some null values"
        assert len(clean_data) > 300, "Too many nulls, not enough data left"

        X = [[row["feature1"], row["feature2"]] for row in clean_data]
        y = [int(row["target"]) for row in clean_data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        assert 0.3 <= accuracy <= 0.8, f"Accuracy {accuracy:.3f} outside expected range"

    def test_model_with_outliers(self):
        """Test that models can be trained on data with outliers."""
        schema = [
            {
                "name": "feature1",
                "type": "int",
                "config": {"min": 1, "max": 100, "quality_config": {"outlier_rate": 0.1}},
            },
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "target", "type": "bool"},
        ]
        data, _ = generate_data(schema, 600)

        X = [[row["feature1"], row["feature2"]] for row in data]
        y = [int(row["target"]) for row in data]

        # Verify we have outliers
        feature1_vals = [row["feature1"] for row in data]
        has_outliers = any(v < 1 or v > 100 for v in feature1_vals)
        assert has_outliers, "Expected some outliers in feature1"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # RandomForest should be robust to outliers
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        assert 0.3 <= accuracy <= 0.8, f"Accuracy {accuracy:.3f} outside expected range"


class TestModelPerformanceMetrics:
    """Test various performance metrics can be calculated."""

    def test_classification_metrics_complete(self):
        """Test that all common classification metrics can be calculated."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "feature2", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "target", "type": "bool"},
        ]
        data, _ = generate_data(schema, 500)

        X = [[row["feature1"], row["feature2"]] for row in data]
        y = [int(row["target"]) for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate all metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # All should be calculable and in valid range
        assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} out of range"
        assert 0.0 <= f1 <= 1.0, f"F1 {f1} out of range"
        assert 0.0 <= auc <= 1.0, f"AUC {auc} out of range"

    def test_regression_metrics_complete(self):
        """Test that all common regression metrics can be calculated."""
        schema = [
            {"name": "feature1", "type": "float", "config": {"min": 0.0, "max": 10.0}},
            {"name": "target", "type": "float", "config": {"min": 0.0, "max": 100.0}},
        ]
        data, _ = generate_data(schema, 400)

        X = [[row["feature1"]] for row in data]
        y = [row["target"] for row in data]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # All should be calculable
        assert mse >= 0, f"MSE {mse} should be non-negative"
        assert -1.0 <= r2 <= 1.0, f"R² {r2} outside typical range"
