"""
End-to-end tests for ExtraTrees to IsolationForest converter.

Run with: uv run pytest test_converter.py -v
"""

import pytest
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import roc_auc_score

from extratrees_to_iforest import convert_extratrees_to_isolationforest


# Fixtures
@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    return X, y


@pytest.fixture
def small_data():
    """Generate small dataset for ONNX tests."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    return X, y


# ExtraTreesClassifier Tests
class TestExtraTreesClassifier:
    """Test conversion of ExtraTreesClassifier."""
    
    def test_basic_conversion(self, classification_data):
        """Test basic classifier conversion."""
        X, y = classification_data
        
        # Train model
        et = ExtraTreesClassifier(n_estimators=10, random_state=42)
        et.fit(X, y)
        
        # Convert
        iforest = convert_extratrees_to_isolationforest(et)
        
        # Verify conversion
        assert iforest is not None
        assert hasattr(iforest, 'estimators_')
        assert len(iforest.estimators_) == 10
    
    def test_predictions(self, classification_data):
        """Test that converted model can make predictions."""
        X, y = classification_data
        
        et = ExtraTreesClassifier(n_estimators=20, random_state=42)
        et.fit(X, y)
        iforest = convert_extratrees_to_isolationforest(et, contamination=0.1)
        
        # Make predictions
        predictions = iforest.predict(X)
        scores = iforest.score_samples(X)
        decisions = iforest.decision_function(X)
        
        # Verify outputs
        assert predictions.shape == (len(X),)
        assert scores.shape == (len(X),)
        assert decisions.shape == (len(X),)
        assert set(predictions).issubset({-1, 1})
        
        # Check that some outliers are detected
        n_outliers = np.sum(predictions == -1)
        assert n_outliers > 0
        assert n_outliers < len(X)
    
    def test_different_parameters(self, classification_data):
        """Test conversion with different parameters."""
        X, y = classification_data
        
        # Different contamination values
        et = ExtraTreesClassifier(n_estimators=15, random_state=42)
        et.fit(X, y)
        
        iforest1 = convert_extratrees_to_isolationforest(et, contamination=0.05)
        iforest2 = convert_extratrees_to_isolationforest(et, contamination=0.2)
        
        # Both should produce predictions
        pred1 = iforest1.predict(X)
        pred2 = iforest2.predict(X)
        
        assert len(pred1) == len(X)
        assert len(pred2) == len(X)
        
        # Verify they work with different offset values
        iforest3 = convert_extratrees_to_isolationforest(et, contamination=0.1, offset=-0.3)
        pred3 = iforest3.predict(X)
        assert len(pred3) == len(X)
    
    def test_with_max_features(self, classification_data):
        """Test conversion with different max_features settings."""
        X, y = classification_data
        
        # Test with different max_features
        for max_features in [None, 'sqrt', 5, 0.5]:
            et = ExtraTreesClassifier(
                n_estimators=10,
                max_features=max_features,
                random_state=42
            )
            et.fit(X, y)
            
            iforest = convert_extratrees_to_isolationforest(et)
            predictions = iforest.predict(X)
            
            assert predictions is not None
            assert len(predictions) == len(X)


# ExtraTreesRegressor Tests
class TestExtraTreesRegressor:
    """Test conversion of ExtraTreesRegressor."""
    
    def test_basic_conversion(self, regression_data):
        """Test basic regressor conversion."""
        X, y = regression_data
        
        # Train model
        et = ExtraTreesRegressor(n_estimators=10, random_state=42)
        et.fit(X, y)
        
        # Convert
        iforest = convert_extratrees_to_isolationforest(et)
        
        # Verify conversion
        assert iforest is not None
        assert hasattr(iforest, 'estimators_')
        assert len(iforest.estimators_) == 10
    
    def test_predictions(self, regression_data):
        """Test that converted regressor can make predictions."""
        X, y = regression_data
        
        et = ExtraTreesRegressor(n_estimators=20, random_state=42)
        et.fit(X, y)
        iforest = convert_extratrees_to_isolationforest(et, contamination='auto')
        
        # Make predictions
        predictions = iforest.predict(X)
        scores = iforest.score_samples(X)
        
        # Verify outputs
        assert predictions.shape == (len(X),)
        assert scores.shape == (len(X),)
        assert set(predictions).issubset({-1, 1})
    
    def test_regressor_with_classification_data(self, classification_data):
        """Test regressor with classification-style targets."""
        X, y = classification_data
        
        et = ExtraTreesRegressor(n_estimators=15, random_state=42)
        et.fit(X, y.astype(float))
        
        iforest = convert_extratrees_to_isolationforest(et)
        predictions = iforest.predict(X)
        
        assert predictions is not None
        assert len(predictions) == len(X)


# Error Handling Tests
class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_unfitted_model_raises_error(self, classification_data):
        """Test that unfitted model raises ValueError."""
        X, y = classification_data
        
        et = ExtraTreesClassifier(n_estimators=10)
        # Don't fit the model
        
        with pytest.raises(ValueError, match="must be fitted"):
            convert_extratrees_to_isolationforest(et)
    
    def test_wrong_model_type_raises_error(self, classification_data):
        """Test that wrong model type raises TypeError."""
        X, y = classification_data
        
        # Use RandomForest instead of ExtraTrees
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        with pytest.raises(TypeError, match="Expected ExtraTreesClassifier or ExtraTreesRegressor"):
            convert_extratrees_to_isolationforest(rf)
    
    def test_none_input_raises_error(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError):
            convert_extratrees_to_isolationforest(None)


# Model Attributes Tests
class TestModelAttributes:
    """Test that converted model has correct attributes."""
    
    def test_estimator_count_preserved(self, classification_data):
        """Test that number of estimators is preserved."""
        X, y = classification_data
        
        for n_estimators in [5, 10, 50]:
            et = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42)
            et.fit(X, y)
            
            iforest = convert_extratrees_to_isolationforest(et)
            
            assert len(iforest.estimators_) == n_estimators
    
    def test_features_preserved(self, classification_data):
        """Test that feature information is preserved."""
        X, y = classification_data
        
        et = ExtraTreesClassifier(n_estimators=10, random_state=42)
        et.fit(X, y)
        
        iforest = convert_extratrees_to_isolationforest(et)
        
        assert iforest.n_features_in_ == X.shape[1]
        assert hasattr(iforest, '_max_features')
        assert hasattr(iforest, '_max_samples')
    
    def test_internal_attributes_set(self, classification_data):
        """Test that internal attributes are properly set."""
        X, y = classification_data
        
        et = ExtraTreesClassifier(n_estimators=10, random_state=42)
        et.fit(X, y)
        
        iforest = convert_extratrees_to_isolationforest(et)
        
        # Check internal attributes required by IsolationForest
        assert hasattr(iforest, 'estimators_features_')
        assert hasattr(iforest, '_decision_path_lengths')
        assert hasattr(iforest, '_average_path_length_per_tree')
        assert hasattr(iforest, 'offset_')


# ONNX Export Tests
class TestONNXExport:
    """Test ONNX export functionality."""
    
    @pytest.mark.parametrize("model_class", [ExtraTreesClassifier, ExtraTreesRegressor])
    def test_onnx_export_with_all_features(self, small_data, model_class):
        """Test ONNX export with max_features=None (all features)."""
        pytest.importorskip("onnx")
        pytest.importorskip("skl2onnx")
        
        from skl2onnx import to_onnx
        import onnx
        
        X, y = small_data
        
        # Train with max_features=None for ONNX compatibility
        if model_class == ExtraTreesClassifier:
            et = model_class(n_estimators=5, max_features=None, random_state=42)
        else:
            et = model_class(n_estimators=5, max_features=None, random_state=42)
        
        et.fit(X, y)
        iforest = convert_extratrees_to_isolationforest(et)
        
        # Export to ONNX
        onnx_model = to_onnx(
            iforest,
            X[:1].astype(np.float32),
            target_opset={"": 15, "ai.onnx.ml": 2}
        )
        
        # Verify ONNX model
        assert onnx_model is not None
        onnx.checker.check_model(onnx_model)
    
    def test_onnx_export_fails_with_feature_subsampling(self, small_data):
        """Test that ONNX export fails with feature subsampling."""
        pytest.importorskip("onnx")
        pytest.importorskip("skl2onnx")
        
        from skl2onnx import to_onnx
        
        X, y = small_data
        
        # Train with max_features != None (feature subsampling)
        et = ExtraTreesClassifier(n_estimators=5, max_features=2, random_state=42)
        et.fit(X, y)
        iforest = convert_extratrees_to_isolationforest(et)
        
        # ONNX export should fail
        with pytest.raises(Exception, match="does not support"):
            to_onnx(
                iforest,
                X[:1].astype(np.float32),
                target_opset={"": 15, "ai.onnx.ml": 2}
            )


# Integration Tests
class TestIntegration:
    """Integration tests combining multiple aspects."""
    
    def test_classifier_end_to_end(self, classification_data):
        """Complete end-to-end test for classifier."""
        X, y = classification_data
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train ExtraTrees
        et = ExtraTreesClassifier(
            n_estimators=50,
            max_features='sqrt',
            random_state=42
        )
        et.fit(X_train, y_train)
        
        # Convert to IsolationForest
        iforest = convert_extratrees_to_isolationforest(et, contamination=0.1)
        
        # Predict on test set
        predictions = iforest.predict(X_test)
        scores = iforest.score_samples(X_test)
        
        # Verify results
        assert len(predictions) == len(X_test)
        assert len(scores) == len(X_test)
        assert np.sum(predictions == -1) > 0  # Some outliers detected
    
    def test_regressor_end_to_end(self, regression_data):
        """Complete end-to-end test for regressor."""
        X, y = regression_data
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train ExtraTrees
        et = ExtraTreesRegressor(
            n_estimators=50,
            max_features=0.8,
            random_state=42
        )
        et.fit(X_train, y_train)
        
        # Convert to IsolationForest
        iforest = convert_extratrees_to_isolationforest(et, contamination='auto')
        
        # Predict on test set
        predictions = iforest.predict(X_test)
        scores = iforest.score_samples(X_test)
        
        # Verify results
        assert len(predictions) == len(X_test)
        assert len(scores) == len(X_test)
        assert -1 in predictions or 1 in predictions  # Has predictions


# Performance Tests
class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_model_conversion(self):
        """Test conversion of large model."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            random_state=42
        )
        
        # Train large model
        et = ExtraTreesClassifier(n_estimators=100, random_state=42)
        et.fit(X, y)
        
        # Convert
        iforest = convert_extratrees_to_isolationforest(et)
        
        # Make predictions (should complete without timeout)
        predictions = iforest.predict(X)
        
        assert len(predictions) == len(X)
    
    def test_score_consistency(self, classification_data):
        """Test that scores are consistent across multiple calls."""
        X, y = classification_data
        
        et = ExtraTreesClassifier(n_estimators=20, random_state=42)
        et.fit(X, y)
        iforest = convert_extratrees_to_isolationforest(et)
        
        # Get scores multiple times
        scores1 = iforest.score_samples(X)
        scores2 = iforest.score_samples(X)
        scores3 = iforest.score_samples(X)
        
        # Scores should be identical
        np.testing.assert_array_equal(scores1, scores2)
        np.testing.assert_array_equal(scores2, scores3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

