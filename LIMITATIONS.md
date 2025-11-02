# Limitations and Known Issues

This document provides a comprehensive overview of all limitations, constraints, and known issues when converting ExtraTreesClassifier/Regressor models to IsolationForest.

## Table of Contents
1. [Pre-conversion Requirements](#pre-conversion-requirements)
2. [ONNX Export Limitations](#onnx-export-limitations)
3. [Memory Considerations](#memory-considerations)
4. [Behavioral Differences](#behavioral-differences)
5. [Performance Considerations](#performance-considerations)
6. [Workarounds and Solutions](#workarounds-and-solutions)

---

## Pre-conversion Requirements

### 1. Model Must Be Fitted
**Issue**: The ExtraTrees model must be fitted before conversion.

**Details**: 
- The conversion function requires access to the `estimators_` attribute
- This attribute is only created after calling `fit()` on the model
- Attempting to convert an unfitted model will raise a `ValueError`

**Example**:
```python
from extratrees_to_iforest import convert_extratrees_to_isolationforest
from sklearn.ensemble import ExtraTreesClassifier

# ❌ WRONG - Will fail
et = ExtraTreesClassifier(n_estimators=10)
iforest = convert_extratrees_to_isolationforest(et)  # ValueError!

# ✅ CORRECT
et = ExtraTreesClassifier(n_estimators=10)
et.fit(X, y)  # Must fit first
iforest = convert_extratrees_to_isolationforest(et)  # Works!
```

### 2. Model Type Validation
**Issue**: Only ExtraTreesClassifier and ExtraTreesRegressor are supported.

**Details**:
- RandomForestClassifier/Regressor are NOT supported
- GradientBoostingClassifier/Regressor are NOT supported
- Other ensemble methods are NOT supported
- Attempting to convert unsupported models will raise a `TypeError`

**Example**:
```python
from sklearn.ensemble import RandomForestClassifier

# ❌ WRONG - RandomForest not supported
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)
iforest = convert_extratrees_to_isolationforest(rf)  # TypeError!
```

---

## ONNX Export Limitations

### 3. Feature Subsampling Incompatibility
**Issue**: ONNX export only works when `max_features=None` (all features used).

**Severity**: ⚠️ HIGH - Prevents ONNX export in common scenarios

**Details**:
- The skl2onnx converter for IsolationForest does not support feature subsampling
- If `_max_features != n_features`, ONNX export will fail
- This affects models trained with:
  - `max_features='sqrt'`
  - `max_features='log2'`
  - `max_features=<int less than n_features>`
  - `max_features=<float less than 1.0>`

**Error Message**:
```
Converter for IsolationForest does not support the case when 
_max_features=X != number of given features Y.
```

**Workaround**:
```python
# For ONNX export, train with max_features=None
et_model = ExtraTreesClassifier(
    n_estimators=100,
    max_features=None,  # Use ALL features for ONNX compatibility
    random_state=42
)
et_model.fit(X, y)
iforest = convert_extratrees_to_isolationforest(et_model)

# Now ONNX export will work
from skl2onnx import to_onnx
onnx_model = to_onnx(iforest, X[:1].astype(np.float32))
```

**Impact**:
- ✅ Can still use the converted model in Python
- ❌ Cannot export to ONNX for production deployment
- ⚠️ May reduce model diversity if forced to use all features

---

## Memory Considerations

### 4. Deep Copy of Tree Structures
**Issue**: The conversion creates deep copies of all tree estimators.

**Severity**: ⚠️ MEDIUM - Can cause memory issues with large models

**Details**:
- Each tree in the ExtraTrees model is deep copied to the IsolationForest
- Memory usage approximately doubles during conversion
- Large models (many estimators or deep trees) can exhaust available memory

**Memory Impact**:
```python
# Example: Model with 1000 trees, each ~1MB
# Original model: ~1GB memory
# During conversion: ~2GB memory (original + copy)
# After conversion: ~2GB memory (both models in memory)
```

**Workarounds**:
1. Delete the original model after conversion if not needed:
```python
iforest = convert_extratrees_to_isolationforest(et_model)
del et_model  # Free up memory
```

2. Use fewer estimators for memory-constrained environments:
```python
et_model = ExtraTreesClassifier(n_estimators=50)  # Instead of 1000
```

3. Convert in batches if processing multiple models

### 5. max_samples Float Values
**Issue**: When `max_samples` is a float, the actual sample count cannot be determined without training data.

**Details**:
- IsolationForest requires `_max_samples` to be an integer
- If the original model has `max_samples=0.5` (50% of data), we don't know the training set size
- The converter defaults to `_max_samples=256` in this case
- This may affect anomaly score calibration

**Workaround**:
```python
# If you know your training set size, set max_samples as an integer
n_samples_train = len(X_train)
max_samples_int = int(0.5 * n_samples_train)

et_model = ExtraTreesClassifier(
    n_estimators=100,
    max_samples=max_samples_int  # Use integer instead of float
)
```

---

## Behavioral Differences

### 6. Different Splitting Criteria
**Issue**: ExtraTrees use different splitting criteria than IsolationForest.

**Severity**: ℹ️ INFORMATIONAL - Expected behavior difference

**Details**:
- **IsolationForest**: Uses completely random splits on random features
- **ExtraTrees**: Uses extremely randomized splits but considers target variable
- The converted model uses ExtraTrees splitting logic, not IsolationForest logic
- Anomaly scores may differ from a natively trained IsolationForest

**Comparison**:
```
IsolationForest (native):
- Unsupervised: Doesn't use labels
- Random split points
- Optimized for isolation

ExtraTrees → IsolationForest (converted):
- Originally supervised: Trained with labels
- Extremely randomized but informed splits
- Trees optimized for classification/regression
```

**Implications**:
- Converted models may perform differently on anomaly detection
- May be better for some use cases (leveraging supervised information)
- May be worse for pure anomaly detection without labeled data

### 7. Anomaly Score Interpretation
**Issue**: Anomaly scores reflect supervised/regression training, not pure isolation.

**Details**:
- Standard IsolationForest scores: Higher = more isolated = more anomalous
- Converted model scores: Based on tree structures trained for different objectives
- Score distributions may differ significantly
- Contamination parameter may need different tuning

**Recommendation**:
- Always validate on a holdout set
- Tune the contamination parameter for your specific use case
- Compare with native IsolationForest if possible

### 8. Feature Importance Not Available
**Issue**: IsolationForest does not have feature_importances_ attribute.

**Details**:
- ExtraTrees models have `feature_importances_` 
- IsolationForest does not support this attribute
- The converted model will not have feature importance information
- Feature importance from the original model may not be relevant for anomaly detection

**Workaround**:
```python
# Save feature importances before conversion
feature_importances = et_model.feature_importances_.copy()

# Then convert
iforest = convert_extratrees_to_isolationforest(et_model)

# Feature importances from original model (may not reflect anomaly relevance)
print(f"Original feature importances: {feature_importances}")
```

---

## Performance Considerations

### 9. Prediction Speed
**Issue**: Prediction speed may differ from native IsolationForest.

**Details**:
- ExtraTrees may have deeper trees than IsolationForest
- ExtraTrees may consider more features at each split
- Prediction time depends on tree depth and complexity
- Generally comparable, but may be slower for very deep trees

**Benchmarking Recommendation**:
```python
import time

# Measure prediction time
start = time.time()
predictions = iforest.predict(X_test)
elapsed = time.time() - start
print(f"Prediction time: {elapsed:.4f}s for {len(X_test)} samples")
```

### 10. Training Data Size Sensitivity
**Issue**: ExtraTrees behavior depends heavily on training data characteristics.

**Details**:
- If ExtraTrees was trained on small dataset, trees may be overly specific
- If trained on large dataset, trees may be too general
- Original training data size affects converted model's anomaly detection
- IsolationForest typically uses subsampling; ExtraTrees may not

**Best Practices**:
- Use `max_samples` parameter when training ExtraTrees
- Train on representative data
- Validate on out-of-sample data

---

## Workarounds and Solutions

### Summary Table

| Limitation | Severity | Workaround Available | Impact |
|------------|----------|---------------------|---------|
| Must be fitted | ⚠️ HIGH | Yes - Call fit() first | Blocks conversion |
| ONNX export max_features | ⚠️ HIGH | Yes - Use max_features=None | Prevents deployment |
| Memory usage | ⚠️ MEDIUM | Partial - Delete original | May exhaust memory |
| Different splitting | ℹ️ INFO | No - By design | Behavior difference |
| max_samples float | ⚠️ LOW | Yes - Use integer | Score calibration |
| Feature importance | ℹ️ INFO | Partial - Save beforehand | Loss of information |

### General Best Practices

1. **For Production Deployment**:
   ```python
   # Train with ONNX-compatible settings
   et_model = ExtraTreesClassifier(
       n_estimators=100,
       max_features=None,      # ONNX compatibility
       max_samples=256,        # Fixed sample count
       bootstrap=True,
       random_state=42
   )
   ```

2. **For Memory-Constrained Environments**:
   ```python
   # Use fewer, shallower trees
   et_model = ExtraTreesClassifier(
       n_estimators=50,        # Fewer trees
       max_depth=10,           # Limit depth
       min_samples_leaf=5      # Prune small leaves
   )
   ```

3. **For Maximum Performance**:
   ```python
   # After conversion, delete original if not needed
   iforest = convert_extratrees_to_isolationforest(et_model)
   del et_model
   import gc
   gc.collect()
   ```

4. **For Validation**:
   ```python
   # Always validate on holdout data
   from sklearn.metrics import roc_auc_score
   
   scores = iforest.score_samples(X_test)
   auc = roc_auc_score(y_test_binary, -scores)
   print(f"Anomaly detection AUC: {auc:.4f}")
   ```

---

## Getting Help

If you encounter issues not listed here:

1. Check if your ExtraTrees model is fitted
2. Verify you're using ExtraTreesClassifier or ExtraTreesRegressor
3. For ONNX issues, ensure `max_features=None`
4. For memory issues, try reducing n_estimators
5. Open an issue on GitHub with:
   - Python version
   - scikit-learn version
   - Minimal reproducible example
   - Full error traceback

---

**Last Updated**: November 2, 2025  
**Version**: 0.1.0

