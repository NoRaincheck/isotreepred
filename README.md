# IsoTreePred: Tree-Based Machine Learning Toolkit

A comprehensive Python library for tree-based machine learning, featuring:
- **Mondrian Forest**: Online random forests with incremental learning support
- **ExtraTrees to IsolationForest Converter**: Repurpose trained models for anomaly detection
- **Full Pipeline Support**: Mondrian Forest → ExtraTrees → IsolationForest

## Overview

This project provides two main components:

### 1. Mondrian Forest (NEW!)

An implementation of online random forests that support incremental learning through `partial_fit`:
- **Online Learning**: Update models with new data without full retraining
- **Scikit-learn Compatible**: Follows standard sklearn API (fit, predict, score, etc.)
- **Export Support**: Convert to ExtraTreesClassifier/ExtraTreesRegressor
- **Both Tasks**: Classification and regression support

### 2. ExtraTrees to IsolationForest Converter

Convert trained ExtraTrees models into IsolationForest for anomaly detection:
- Pre-trained ExtraTrees models you want to use for anomaly detection
- Leverage tree structures from supervised learning for unsupervised outlier detection
- Export ExtraTrees-based anomaly detection to ONNX format

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd isotreepred

# Install dependencies using uv
uv sync
```

## Requirements

- Python >= 3.10
- scikit-learn >= 1.7.2
- numpy >= 2.2.6
- skl2onnx >= 1.19.1 (for ONNX export)

## Quick Start

### Mondrian Forest (Online Learning)

```python
from mondrian_forest import MondrianForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Train with initial batch
mf = MondrianForestClassifier(n_estimators=100, random_state=42)
mf.fit(X[:500], y[:500])

# Update with new data (online learning!)
mf.partial_fit(X[500:], y[500:])

# Make predictions
predictions = mf.predict(X)
probabilities = mf.predict_proba(X)
```

### ExtraTrees to IsolationForest Converter

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from extratrees_to_iforest import convert_extratrees_to_isolationforest

# Train an ExtraTreesClassifier
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X, y)

# Convert to IsolationForest
iforest = convert_extratrees_to_isolationforest(et_model, contamination=0.1)

# Use for anomaly detection
predictions = iforest.predict(X)  # Returns 1 for inliers, -1 for outliers
scores = iforest.score_samples(X)  # Returns anomaly scores
```

### Full Pipeline: Mondrian Forest → ExtraTrees → IsolationForest

```python
from mondrian_forest import MondrianForestClassifier, export_mondrian_to_extratrees
from extratrees_to_iforest import convert_extratrees_to_isolationforest

# Train Mondrian Forest
mf = MondrianForestClassifier(n_estimators=50, random_state=42)
mf.fit(X, y)

# Export to ExtraTrees
et = export_mondrian_to_extratrees(mf)

# Convert to IsolationForest for anomaly detection
iforest = convert_extratrees_to_isolationforest(et, contamination=0.1)
anomalies = iforest.predict(X)
```

## Mondrian Forest Usage

### Classification with Online Learning

```python
from mondrian_forest import MondrianForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train Mondrian Forest
mf = MondrianForestClassifier(n_estimators=50, random_state=42)
mf.fit(X_train, y_train)

# Make predictions
y_pred = mf.predict(X_test)
y_proba = mf.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Online learning - update with new data
new_X, new_y = make_classification(n_samples=100, n_features=10, random_state=43)
mf.partial_fit(new_X, new_y)
```

### Regression with Incremental Updates

```python
from mondrian_forest import MondrianForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

# Generate data
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)

# Train in batches (simulating streaming data)
mf = MondrianForestRegressor(n_estimators=50, random_state=42)

# Initial training
mf.fit(X[:500], y[:500])

# Incremental updates
for i in range(5, 10):
    start = i * 100
    end = start + 100
    mf.partial_fit(X[start:end], y[start:end])

# Make predictions
predictions = mf.predict(X)
r2 = r2_score(y, predictions)
print(f"R² score: {r2:.3f}")
```

### Export to ExtraTrees

```python
from mondrian_forest import MondrianForestClassifier, export_mondrian_to_extratrees

# Train Mondrian Forest
mf = MondrianForestClassifier(n_estimators=30, random_state=42)
mf.fit(X_train, y_train)

# Export to ExtraTrees format
et = export_mondrian_to_extratrees(mf)

# Use as a standard ExtraTrees model
et_predictions = et.predict(X_test)
```

### Lifetime Parameter

The `lifetime` parameter controls tree depth. Shorter lifetimes create shallower trees:

```python
# Shallow trees (faster, less accurate)
mf_shallow = MondrianForestClassifier(
    n_estimators=50, 
    lifetime=1.0,
    random_state=42
)

# Deep trees (slower, more accurate)
mf_deep = MondrianForestClassifier(
    n_estimators=50,
    lifetime=100.0,
    random_state=42
)

# No constraint (default)
mf_unconstrained = MondrianForestClassifier(
    n_estimators=50,
    lifetime=np.inf,
    random_state=42
)
```

### Run Complete Examples

```bash
# Run all Mondrian Forest examples
uv run python mondrian_example.py
```

## Sample Usage

### Basic Anomaly Detection

```python
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from extratrees_to_iforest import convert_extratrees_to_isolationforest

# Generate dataset
X, y = make_classification(
    n_samples=500, 
    n_features=10, 
    n_informative=5,
    random_state=42
)

# Train ExtraTrees model
et_model = ExtraTreesClassifier(
    n_estimators=50,
    max_features='sqrt',
    random_state=42
)
et_model.fit(X, y)

# Convert to IsolationForest
iforest = convert_extratrees_to_isolationforest(
    et_model,
    contamination=0.1  # Expect 10% outliers
)

# Detect anomalies
predictions = iforest.predict(X)
scores = iforest.score_samples(X)

# Analyze results
outliers = X[predictions == -1]
inliers = X[predictions == 1]

print(f"Total samples: {len(X)}")
print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(X)*100:.1f}%)")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
```

### Using with ExtraTreesRegressor

```python
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import make_regression

# Generate regression dataset
X, y = make_regression(n_samples=500, n_features=10, random_state=42)

# Train regressor
et_reg = ExtraTreesRegressor(
    n_estimators=50,
    max_features=0.8,
    random_state=42
)
et_reg.fit(X, y)

# Convert to IsolationForest
iforest = convert_extratrees_to_isolationforest(et_reg, contamination='auto')

# Detect anomalies
predictions = iforest.predict(X)
print(f"Outliers detected: {np.sum(predictions == -1)}")
```

### Train-Test Split Example

```python
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train on training set
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)

# Convert to IsolationForest
iforest = convert_extratrees_to_isolationforest(et_model, contamination=0.15)

# Detect anomalies on test set
test_predictions = iforest.predict(X_test)
test_scores = iforest.score_samples(X_test)

print(f"Test set outliers: {np.sum(test_predictions == -1)}/{len(X_test)}")
```

### ONNX Export Example

```python
from skl2onnx import to_onnx
import onnx

# Train with max_features=None for ONNX compatibility
et_model = ExtraTreesClassifier(
    n_estimators=20,
    max_features=None,  # Required for ONNX export
    random_state=42
)
et_model.fit(X, y)

# Convert to IsolationForest
iforest = convert_extratrees_to_isolationforest(et_model)

# Export to ONNX
onnx_model = to_onnx(
    iforest,
    X[:1].astype(np.float32),
    target_opset={"": 15, "ai.onnx.ml": 2}
)

# Save ONNX model
onnx.save_model(onnx_model, 'isolation_forest.onnx')
print("Model exported to ONNX format")
```

## API Reference

### Function Signature

```python
def convert_extratrees_to_isolationforest(
    extratrees_model, 
    contamination='auto', 
    offset=None
)
```

### Parameters

- **extratrees_model**: `ExtraTreesClassifier` or `ExtraTreesRegressor`
  - A fitted ExtraTreesClassifier or ExtraTreesRegressor model to convert
  - Must be already trained (fitted) before conversion

- **contamination**: `'auto'` or `float`, default=`'auto'`
  - The proportion of outliers in the dataset
  - If `'auto'`, the threshold is determined as in the original IsolationForest
  - If float, should be in the range (0, 0.5]

- **offset**: `float`, optional
  - Offset used to define the decision function
  - If `None`, computed automatically based on contamination parameter
  - Default is -0.5 for contamination='auto'

### Returns

- **iforest**: `IsolationForest`
  - A new IsolationForest object with tree structures copied from the input model
  - Can be used like any standard IsolationForest for anomaly detection

### Examples

#### Example 1: Basic Conversion

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from extratrees_to_iforest import convert_extratrees_to_isolationforest

# Create and train model
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
et = ExtraTreesClassifier(n_estimators=50, random_state=42)
et.fit(X, y)

# Convert to IsolationForest
iforest = convert_extratrees_to_isolationforest(et)

# Detect anomalies
outliers = iforest.predict(X)
print(f"Found {sum(outliers == -1)} outliers")
```

#### Example 2: Using ExtraTreesRegressor

```python
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import make_regression
from extratrees_to_iforest import convert_extratrees_to_isolationforest

# Create and train regressor
X, y = make_regression(n_samples=500, n_features=10, random_state=42)
et_reg = ExtraTreesRegressor(n_estimators=100, max_features=0.8, random_state=42)
et_reg.fit(X, y)

# Convert to IsolationForest with custom contamination
iforest = convert_extratrees_to_isolationforest(et_reg, contamination=0.05)

# Get anomaly scores
scores = iforest.score_samples(X)
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
```

#### Example 3: Running All Examples

```bash
# Run the comprehensive example file
uv run python conversion_example.py
```

This will demonstrate:
- Converting ExtraTreesClassifier to IsolationForest
- Converting ExtraTreesRegressor to IsolationForest
- Comparing different model configurations

## Limitations ⚠️

**IMPORTANT**: Please review all limitations before using in production.

### Quick Summary

1. **Model must be fitted** before conversion (call `fit()` first)
2. **ONNX export** only works with `max_features=None` (all features)
3. **Memory usage** approximately doubles during conversion (deep copy of trees)
4. **Different behavior** from native IsolationForest (uses ExtraTrees splits, not random splits)
5. **Only supports** ExtraTreesClassifier and ExtraTreesRegressor (not RandomForest or others)

### Detailed Documentation

For comprehensive information about all limitations, constraints, and workarounds, see:

📖 **[LIMITATIONS.md](LIMITATIONS.md)** - Complete limitations documentation

This document covers:
- Pre-conversion requirements
- ONNX export limitations and workarounds
- Memory considerations
- Behavioral differences from native IsolationForest
- Performance considerations
- Best practices and solutions

## How It Works

The conversion process involves several steps:

1. **Parameter Extraction**: Extracts relevant parameters from the ExtraTrees model (n_estimators, max_features, max_samples, etc.)

2. **Tree Structure Copy**: Deep copies all tree estimators from the ExtraTrees model to ensure independence

3. **Feature Mapping**: Preserves which features each estimator uses (estimators_features_)

4. **Path Length Computation**: Pre-computes decision path lengths and average path lengths for each tree, which are required for IsolationForest's anomaly scoring

5. **Attribute Setup**: Sets all internal attributes required by IsolationForest (_max_features, _max_samples, offset_, etc.)

The resulting IsolationForest object can be used exactly like a standard sklearn IsolationForest, with full support for:
- `predict(X)`: Predict if samples are outliers or inliers
- `score_samples(X)`: Compute anomaly scores
- `decision_function(X)`: Compute decision function values
- ONNX export via skl2onnx

## Technical Details

### Internal Attributes Set

The function sets the following internal attributes required by IsolationForest:

- `estimators_`: Deep copies of decision trees
- `estimators_features_`: Feature indices used by each estimator
- `n_features_in_`: Number of input features
- `_max_features`: Maximum number of features to use
- `_max_samples`: Maximum number of samples for training
- `max_samples_`: Public version of max samples
- `_decision_path_lengths`: Pre-computed node depths
- `_average_path_length_per_tree`: Pre-computed average path lengths
- `offset_`: Decision function offset

### Differences from Standard IsolationForest

While the converted model behaves like an IsolationForest, there are some differences:

1. **Tree Structure**: The trees come from ExtraTrees (which uses different splitting criteria) rather than IsolationForest's random splits

2. **Feature Selection**: ExtraTrees may use different feature subsampling than standard IsolationForest

3. **Scores**: Anomaly scores will be based on the ExtraTrees structure, which was trained on labeled data (for classifiers) or continuous targets (for regressors)

## Running Tests

The project includes comprehensive end-to-end tests using pytest.

```bash
# Run all tests
uv run pytest -v

# Run only Mondrian Forest tests
uv run pytest test_mondrian_forest.py -v

# Run only converter tests
uv run pytest test_converter.py -v

# Run specific test class
uv run pytest test_mondrian_forest.py::TestMondrianForestClassifier -v

# Run with coverage
uv run pytest --cov=mondrian_forest --cov=extratrees_to_iforest

# Run only ONNX tests (requires onnx and skl2onnx)
uv run pytest test_converter.py::TestONNXExport -v
```

### Test Coverage

The test suite includes:

**Mondrian Forest Tests** (28 tests):
- ✅ **Classifier**: Basic fit, predict, predict_proba, score
- ✅ **Regressor**: Basic fit, predict, score
- ✅ **Online Learning**: partial_fit (initial and incremental)
- ✅ **Export**: Mondrian Forest → ExtraTrees conversion
- ✅ **Integration**: Full pipeline (MF → ET → IF)
- ✅ **Error handling**: Validation and edge cases
- ✅ **Performance**: Large datasets, prediction consistency

**Converter Tests** (20 tests):
- ✅ **ExtraTreesClassifier** conversion and predictions
- ✅ **ExtraTreesRegressor** conversion and predictions  
- ✅ **ONNX export** compatibility (with max_features=None)
- ✅ **Error handling** (unfitted models, wrong types)
- ✅ **Model attributes** preservation
- ✅ **Integration tests** (end-to-end workflows)
- ✅ **Performance tests** (large models, consistency)

## ONNX Export

The converted IsolationForest can be exported to ONNX format:

```python
from skl2onnx import to_onnx
import numpy

# After conversion
iforest = convert_extratrees_to_isolationforest(et_model)

# Export to ONNX
onnx_model = to_onnx(
    iforest, 
    X[:1].astype(numpy.float32),
    target_opset={"": 15, "ai.onnx.ml": 2}
)

# Save
import onnx
onnx.save_model(onnx_model, 'isolation_forest.onnx')
```

### ONNX Export Limitations

⚠️ **Important**: Due to limitations in the skl2onnx converter, ONNX export currently only works when `max_features=None` (i.e., using all features). If your ExtraTrees model uses feature subsampling (e.g., `max_features='sqrt'` or `max_features=0.8`), the ONNX export will fail.

**Workaround**: Train your ExtraTrees model with `max_features=None`:

```python
# For ONNX export compatibility
et_model = ExtraTreesClassifier(
    n_estimators=100, 
    max_features=None,  # Use all features for ONNX compatibility
    random_state=42
)
```

See `test_converter.py` for complete examples including ONNX export tests.

## Project Structure

```
isotreepred/
├── mondrian_forest.py           # ⭐ NEW - Mondrian Forest implementation
├── extratrees_to_iforest.py    # ⭐ PUBLIC API - Main conversion module
├── test_mondrian_forest.py     # E2E tests for Mondrian Forest (28 tests)
├── test_converter.py            # E2E tests for converter (20 tests)
├── mondrian_example.py          # Example usage demonstrations
├── README.md                    # This file (quick start & usage)
├── LIMITATIONS.md              # Detailed limitations documentation
├── pyproject.toml              # Project configuration
└── uv.lock                     # Locked dependencies
```

### File Descriptions

- **`mondrian_forest.py`** - ⭐ NEW - Mondrian Forest classifier and regressor with online learning
- **`extratrees_to_iforest.py`** - ⭐ Public API module containing the conversion function
- **`test_mondrian_forest.py`** - Comprehensive tests for Mondrian Forest (28 tests)
- **`test_converter.py`** - Comprehensive tests for converter (20 tests)
- **`mondrian_example.py`** - Demonstrates all Mondrian Forest features
- **`README.md`** - Project overview, quick start, and usage examples
- **`LIMITATIONS.md`** - Comprehensive limitations documentation with workarounds

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this converter in your research, please cite:

```bibtex
@software{isotreepred,
  title={IsoTreePred: ExtraTrees to IsolationForest Converter},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

