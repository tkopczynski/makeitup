# Data Generation Tool - Verification Plan

## Overview

This document outlines the comprehensive verification strategy for the data generation tool, covering both general data quality verification and ML training use case validation.

---

## Goals

1. **Ensure data quality**: Verify generated data matches configured specifications
2. **Statistical accuracy**: Confirm quality rates (nulls, duplicates) match targets
3. **ML readiness**: Validate data is suitable for machine learning model training
4. **Performance benchmarking**: Verify models can learn from generated data
5. **Regression prevention**: Detect quality degradation over time

---

## Verification Approaches

### 1. Statistical Validation Suite (Foundation)

Create automated tests that verify the generated data matches expected statistical properties.

#### Components:
- **Distribution checks**: Verify numeric fields follow expected distributions (uniform, normal, etc.)
- **Cardinality tests**: Check if category fields have the right number of unique values
- **Range validation**: Ensure all values fall within configured min/max bounds
- **Quality rate verification**: For messy data, confirm actual null/duplicate/error rates match configured rates within acceptable tolerance (e.g., ±2-3%)
- **Variance checks**: Ensure features aren't constant
- **Uniqueness validation**: Verify unique fields (UUIDs, IDs) have no duplicates

#### Implementation:
- Location: `tests/test_statistical_validation.py`
- Run 1000+ row generations and verify statistical properties
- Use tolerance ranges (e.g., 10% ± 3% for null rates)

---

### 2. Type Conformance Testing

Verify each field type generates valid data according to its specification.

#### Components:
- **Format validation**: Regex checks for emails, phones, UUIDs, dates
- **Type checks**: Ensure int fields are integers, bool fields are booleans, etc.
- **Reference integrity**: Verify all reference values exist in parent tables
- **Date logic**: Confirm dates fall within specified ranges, datetimes have valid components
- **Category membership**: All category values come from allowed list
- **Numeric ranges**: All numeric values within min/max bounds

#### Implementation:
- Location: Extend `tests/test_generator.py`
- Add regex validators for each format type
- Test edge cases (empty strings, nulls, boundary values)

---

### 3. Schema Inference Accuracy Testing

Test how well the LLM converts natural language to schemas.

#### Components:
- **Benchmark dataset**: Create 20-30 natural language descriptions with expected schemas
- **Comparison testing**: Run inference and compare generated schemas to expected ones
- **Quality extraction**: Test if quality parameters ("10% nulls") are correctly parsed
- **Edge cases**: Test ambiguous requests, complex multi-table scenarios
- **Accuracy scoring**: Calculate precision/recall for schema field detection

#### Implementation:
- Location: `tests/test_schema_inference_accuracy.py`
- Store golden test cases as JSON/YAML
- Automated comparison with tolerance for minor variations

#### Example Test Cases:
```yaml
- description: "Generate 1000 customer records with messy emails (10% null, 5% duplicates)"
  expected_schema:
    - name: id
      type: int
    - name: email
      type: email
      config:
        quality_config:
          null_rate: 0.1
          duplicate_rate: 0.05
```

---

### 4. End-to-End Integration Tests

Full pipeline verification from natural language to CSV output.

#### Components:
- **Complete workflows**: Natural language → schema inference → data generation → CSV output
- **Multi-table scenarios**: Test generating related tables with foreign keys
- **Data volume**: Test generation at different scales (100, 1K, 10K, 100K rows)
- **Performance benchmarks**: Track generation speed and memory usage
- **File output validation**: Verify CSV files are properly formatted

#### Implementation:
- Location: `tests/test_integration.py`
- Use temporary directories for file outputs
- Measure and track execution time

---

### 5. Data Quality Profiling

Use data profiling tools to analyze generated output and create quality reports.

#### Components:
- **Automated profiling**: Generate reports showing distributions, nulls, duplicates, outliers
- **Visual inspection**: Create histograms, box plots to spot anomalies (optional)
- **Comparison to real data**: If you have real datasets, compare statistical properties
- **Quality scorecard**: Overall data quality score based on multiple metrics

#### Implementation:
- Location: `src/data_generation/utils/profiler.py`
- Functions:
  - `profile_dataframe(df, schema) -> ProfileReport`
  - `validate_quality_statistics(data, schema) -> QualityReport`
  - `generate_quality_report(data, schema, format='json'|'text')`

#### Report Contents:
```python
{
  "column_name": {
    "configured_null_rate": 0.1,
    "actual_null_rate": 0.103,
    "null_count": 103,
    "configured_duplicate_rate": 0.05,
    "actual_duplicate_rate": 0.048,
    "duplicate_count": 48,
    "unique_count": 849,
    "mean": 42.5,  # for numeric fields
    "std": 15.2,
    "min": 10,
    "max": 75
  }
}
```

---

### 6. Manual Spot Checks

Human verification for realism and coherence.

#### Components:
- **Sample inspection**: Manually review random samples of generated data
- **Domain expert review**: Have someone familiar with the domain assess realism
- **Coherence checks**: Look for logical inconsistencies (e.g., age doesn't match birthdate)
- **Visual review**: Export to CSV and open in spreadsheet for inspection

#### Process:
- Generate 100-500 row samples
- Review for obvious errors
- Check edge cases and boundary values
- Document any issues found

---

### 7. Regression Testing

Prevent quality degradation over time.

#### Components:
- **Golden datasets**: Save known-good outputs and compare new generations
- **Snapshot testing**: Store schemas and verify they remain stable for same inputs
- **Version comparison**: Track changes in output quality across code versions
- **Statistical comparison**: Compare distributions between versions

#### Implementation:
- Location: `tests/test_regression.py`
- Store golden datasets in `tests/fixtures/golden/`
- Use pytest snapshot plugin or custom comparison
- Run on every commit/PR

---

## ML-Specific Verification (New!)

Additional verification needed when data is used for machine learning model training.

### 8. Class Balance Validation

For classification tasks, verify target variable distribution.

#### Components:
- **Target variable distribution**: Is it balanced or intentionally imbalanced?
- **Stratification tests**: Can data be split into train/test/val while maintaining class proportions?
- **Minority class representation**: Do rare classes have enough samples (minimum 30-50)?
- **Class separation**: Classes should be separable but not trivially

#### Implementation:
- Location: `tests/test_ml_validation.py`
- Test functions:
  - `test_class_balance()`
  - `test_stratified_split_feasibility()`
  - `test_minimum_samples_per_class()`

#### Example:
```python
def test_class_balance():
    """Test binary classification has reasonable class balance"""
    schema = [..., {"name": "is_fraud", "type": "bool"}]
    data = generate_data(schema, 1000)

    fraud_rate = sum(row['is_fraud'] for row in data) / len(data)
    assert 0.05 <= fraud_rate <= 0.5  # Between 5% and 50%
```

---

### 9. Feature Correlation & Informativeness

Verify that features are actually useful for prediction.

#### Components:
- **Feature-target correlation**: Features should have some relationship to the target
- **Feature independence**: Check for multicollinearity (VIF < 10)
- **Information leakage detection**: Ensure no features perfectly predict the target (correlation < 0.95)
- **Variance checks**: Features shouldn't be constant (std > 0)
- **Discriminative power**: Features should help separate classes

#### Implementation:
- Location: `tests/test_ml_validation.py`
- Use pandas correlation, sklearn VIF
- Test functions:
  - `test_feature_target_correlation()`
  - `test_no_perfect_predictors()`
  - `test_multicollinearity()`

---

### 10. Data Splitting Validation

Ensure generated data can be properly split for ML training.

#### Components:
- **Train/test separability**: Data can be split temporally or randomly
- **No data leakage**: Related records don't span train/test (check reference types)
- **Sufficient samples per split**: Each split has enough data
- **Stratification preservation**: Class balance maintained in splits

#### Implementation:
- Test 70/30, 80/20, 60/20/20 splits
- Verify stratified splits maintain class balance
- Check reference integrity within splits

---

### 11. Model Performance Benchmarking (Ultimate Test!)

Actually train models on the generated data to verify it's useful.

#### Components:
- **Baseline model training**: Train simple models (LogisticRegression, RandomForest, XGBoost)
- **Performance metrics**: Achieve reasonable accuracy/F1/AUC (not perfect, not random)
  - Binary classification: AUC 0.6-0.9 (better than random 0.5, not perfect 1.0)
  - Regression: R² 0.3-0.9
- **Overfitting checks**: Train/test gap < 0.15
- **Learning curves**: Models should improve with more data

#### Implementation:
- Location: `tests/test_model_training.py`
- Dependencies: scikit-learn, xgboost (optional)
- Test scenarios:
  - Binary classification
  - Multi-class classification
  - Regression

#### Example:
```python
def test_binary_classification_model():
    """Test that a simple model can learn from generated data"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # Generate data with target variable
    schema = [
        {"name": "feature1", "type": "float"},
        {"name": "feature2", "type": "int"},
        {"name": "target", "type": "bool"}
    ]
    data = generate_data(schema, 1000)

    # Convert to numpy arrays
    X = [[row['feature1'], row['feature2']] for row in data]
    y = [row['target'] for row in data]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    # Should be better than random but not perfect
    assert 0.6 <= auc <= 0.95, f"AUC {auc} outside expected range"
```

---

### 12. Scenario-Based Generation Tests

For ML, you often need specific scenarios and controlled experiments.

#### Components:
- **Edge cases**: Rare but important feature combinations
- **Controlled experiments**: Data with known patterns to test model behavior
- **Adversarial examples**: Challenging cases the model should handle
- **Class imbalance scenarios**: Test with 1%, 5%, 10%, 30% minority class
- **Missing data scenarios**: Test with 10%, 25%, 50% missing values

#### Implementation:
- Create predefined scenario templates
- Test model behavior on each scenario
- Verify scenarios generate expected patterns

---

### 13. Target Variable Generation (New Feature!)

This is NEW functionality needed for ML use cases.

#### Design:
Add `target_config` to schema to define how target variables are generated:

```yaml
- name: is_fraud
  type: bool
  config:
    target_config:
      generation_type: "rule_based"  # or "probabilistic"
      rules:
        - condition: "amount > 1000 AND hour >= 22"
          probability: 0.8  # 80% fraud when condition true
        - condition: "num_transactions > 10"
          probability: 0.6
      default_probability: 0.05  # 5% fraud otherwise
```

#### Implementation:
- Location: `src/data_generation/core/target_generation.py`
- Support rule-based and probabilistic targets
- Allow controlled noise/uncertainty
- Multi-class support

#### Verification:
- Test rule application correctness
- Verify probabilities match configuration
- Check class balance matches expectations

---

### 14. ML-Focused Data Profiler

Extend profiler with ML-specific metrics.

#### Components:
- **Class distribution reports**: Per-class statistics
- **Feature correlation matrices**: Heatmap-style correlation analysis
- **Missing value analysis per class**: Are nulls balanced across classes?
- **Feature importance estimates**: Using simple models
- **Data quality scorecard**: ML readiness score (0-100)

#### ML Readiness Scorecard:
```python
{
  "overall_score": 85,  # 0-100
  "class_balance_score": 90,  # Is target balanced?
  "feature_quality_score": 85,  # Are features informative?
  "completeness_score": 80,  # Missing data %
  "consistency_score": 90,  # Data consistency
  "leakage_risk_score": 95,  # Low risk = high score
  "sample_size_score": 100,  # Enough samples?
  "recommendations": [
    "Consider balancing classes (currently 8% minority)",
    "Feature 'X' has high correlation with target (0.98) - possible leakage"
  ]
}
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create statistical validation test suite
- [ ] Implement type conformance tests
- [ ] Add basic data profiler
- [ ] Set up regression testing framework

### Phase 2: ML Basics (Week 2)
- [ ] Implement class balance validation
- [ ] Add feature correlation tests
- [ ] Create data splitting validation
- [ ] Build ML-focused profiler

### Phase 3: Model Benchmarking (Week 3)
- [ ] Implement model training tests
- [ ] Add performance benchmarking
- [ ] Create scenario-based tests
- [ ] Document ML readiness criteria

### Phase 4: Advanced Features (Week 4)
- [ ] Add target variable generation
- [ ] Implement schema inference accuracy tests
- [ ] Create end-to-end integration tests
- [ ] Build quality scorecard system

### Phase 5: Documentation & Polish (Week 5)
- [ ] Document all verification approaches
- [ ] Create example notebooks
- [ ] Add CLI `--verify` flag for quality reports
- [ ] Write user guide for ML use cases

---

## Testing Strategy Summary

### Continuous Tests (Run on every commit)
- Unit tests for all data types
- Statistical validation (quick version: 100 rows)
- Type conformance tests
- Regression tests against golden datasets

### Integration Tests (Run on PR)
- End-to-end pipeline tests
- Schema inference accuracy
- Multi-table generation
- Performance benchmarks

### ML Validation Tests (Run on PR)
- Class balance validation
- Feature correlation tests
- Data splitting tests
- Quick model training (small datasets)

### Comprehensive Tests (Run weekly/release)
- Full model benchmarking (large datasets)
- Extensive scenario testing
- Manual spot checks
- Performance profiling

---

## Success Criteria

### Data Quality
- ✅ 95%+ tests passing
- ✅ Quality rates within ±3% of configured values
- ✅ No type conformance errors
- ✅ Reference integrity 100%

### ML Readiness
- ✅ Models achieve 0.6+ AUC (classification) or 0.3+ R² (regression)
- ✅ No perfect predictors (correlation < 0.95)
- ✅ Sufficient samples per class (min 30)
- ✅ Train/test gap < 0.15

### Performance
- ✅ Generate 1K rows in < 5 seconds
- ✅ Generate 100K rows in < 2 minutes
- ✅ Memory usage < 500MB for 100K rows

---

## Tools & Dependencies

### Required
- pytest (testing framework)
- pandas (data manipulation)
- scikit-learn (ML validation)
- numpy (numerical operations)

### Optional
- xgboost (advanced model testing)
- matplotlib/seaborn (visualization)
- pandas-profiling (automated profiling)
- great_expectations (data validation)

---

## Future Enhancements

### Advanced Verification
- Automated anomaly detection in generated data
- Comparison to real-world dataset distributions
- Privacy preservation verification (k-anonymity, l-diversity)
- Fairness metrics (demographic parity, equal opportunity)

### ML-Specific
- Time series validation (autocorrelation, stationarity)
- NLP-specific validation (text coherence, vocabulary richness)
- Image data validation (pixel distributions, class balance)
- Federated learning scenario support

### Tooling
- Web dashboard for quality reports
- Automated quality alerts
- A/B testing framework for generator improvements
- Integration with MLOps tools (MLflow, Weights & Biases)

---

## References

- Statistical testing: scipy.stats, statsmodels
- ML validation: scikit-learn metrics
- Data profiling: pandas-profiling, ydata-profiling
- Data quality: Great Expectations, Deequ

---

## Notes

This verification plan is designed to be implemented incrementally. Start with Phase 1 (foundation) and expand based on your specific needs and use cases.

The ML-specific verification is crucial because it validates not just that the data is "correct" but that it's actually **useful** for training models. The ultimate test is whether a model can learn meaningful patterns from the generated data.
