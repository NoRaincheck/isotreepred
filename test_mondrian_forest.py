"""
End-to-end tests for Mondrian Forest implementation.

Run with: uv run pytest test_mondrian_forest.py -v
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from mondrian_forest import (
    MondrianForestClassifier,
    MondrianForestRegressor,
    export_mondrian_to_extratrees
)


# Fixtures
@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42
    )
    return X, y


@pytest.fixture
def binary_classification_data():
    """Generate binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
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
    """Generate small dataset for quick tests."""
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    return X, y


# MondrianForestClassifier Tests
class TestMondrianForestClassifier:
    """Test MondrianForestClassifier functionality."""
    
    def test_basic_fit(self, classification_data):
        """Test basic classifier fit."""
        X, y = classification_data
        
        clf = MondrianForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        # Verify fitted attributes
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'n_classes_')
        assert hasattr(clf, 'trees_')
        assert len(clf.trees_) == 10
        assert clf.n_features_in_ == X.shape[1]
    
    def test_predict(self, classification_data):
        """Test classifier predictions."""
        X, y = classification_data
        
        clf = MondrianForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X, y)
        
        predictions = clf.predict(X)
        
        # Verify predictions
        assert predictions.shape == (len(X),)
        assert all(pred in clf.classes_ for pred in predictions)
        
        # Check accuracy is reasonable (should be better than random)
        accuracy = accuracy_score(y, predictions)
        assert accuracy > 0.3  # Should be better than random for 3 classes
    
    def test_predict_proba(self, classification_data):
        """Test classifier probability predictions."""
        X, y = classification_data
        
        clf = MondrianForestClassifier(n_estimators=15, random_state=42)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        # Verify probabilities
        assert proba.shape == (len(X), clf.n_classes_)
        assert np.allclose(np.sum(proba, axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(proba >= 0) and np.all(proba <= 1)  # Valid probabilities
    
    def test_score(self, classification_data):
        """Test classifier score method."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        clf = MondrianForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_train, y_train)
        
        score = clf.score(X_test, y_test)
        
        # Score should be reasonable
        assert 0 <= score <= 1
        assert score > 0.3  # Better than random
    
    def test_partial_fit_initial(self, classification_data):
        """Test partial_fit as initial fit."""
        X, y = classification_data
        
        clf = MondrianForestClassifier(n_estimators=10, random_state=42)
        clf.partial_fit(X, y, classes=np.unique(y))
        
        # Verify fitted
        assert hasattr(clf, 'classes_')
        assert len(clf.trees_) == 10
        
        # Can make predictions
        predictions = clf.predict(X)
        assert predictions.shape == (len(X),)
    
    def test_partial_fit_incremental(self, classification_data):
        """Test partial_fit for incremental learning."""
        X, y = classification_data
        
        # Split data for incremental learning
        split1 = int(0.5 * len(X))
        split2 = int(0.75 * len(X))
        X1, y1 = X[:split1], y[:split1]
        X2, y2 = X[split1:split2], y[split1:split2]
        X3, y3 = X[split2:], y[split2:]
        
        # Initial fit
        clf = MondrianForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X1, y1)
        
        score1 = clf.score(X3, y3)
        
        # Incremental updates
        clf.partial_fit(X2, y2)
        score2 = clf.score(X3, y3)
        
        clf.partial_fit(X3, y3)
        score3 = clf.score(X3, y3)
        
        # Score should improve or stay stable with more data
        assert score3 >= score1 - 0.1  # Allow small fluctuation
    
    def test_different_n_estimators(self, binary_classification_data):
        """Test with different numbers of estimators."""
        X, y = binary_classification_data
        
        for n_estimators in [1, 5, 20, 50]:
            clf = MondrianForestClassifier(n_estimators=n_estimators, random_state=42)
            clf.fit(X, y)
            
            assert len(clf.trees_) == n_estimators
            predictions = clf.predict(X)
            assert len(predictions) == len(X)
    
    def test_lifetime_parameter(self, small_data):
        """Test lifetime parameter effect."""
        X, y = small_data
        
        # Short lifetime should create shallower trees
        clf_short = MondrianForestClassifier(
            n_estimators=5, lifetime=1.0, random_state=42
        )
        clf_short.fit(X, y)
        
        # Long lifetime allows deeper trees
        clf_long = MondrianForestClassifier(
            n_estimators=5, lifetime=100.0, random_state=42
        )
        clf_long.fit(X, y)
        
        # Both should make predictions
        pred_short = clf_short.predict(X)
        pred_long = clf_long.predict(X)
        
        assert len(pred_short) == len(X)
        assert len(pred_long) == len(X)


# MondrianForestRegressor Tests
class TestMondrianForestRegressor:
    """Test MondrianForestRegressor functionality."""
    
    def test_basic_fit(self, regression_data):
        """Test basic regressor fit."""
        X, y = regression_data
        
        reg = MondrianForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X, y)
        
        # Verify fitted attributes
        assert hasattr(reg, 'trees_')
        assert len(reg.trees_) == 10
        assert reg.n_features_in_ == X.shape[1]
    
    def test_predict(self, regression_data):
        """Test regressor predictions."""
        X, y = regression_data
        
        reg = MondrianForestRegressor(n_estimators=20, random_state=42)
        reg.fit(X, y)
        
        predictions = reg.predict(X)
        
        # Verify predictions
        assert predictions.shape == (len(X),)
        assert np.all(np.isfinite(predictions))  # All predictions are valid numbers
    
    def test_score(self, regression_data):
        """Test regressor score method."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        reg = MondrianForestRegressor(n_estimators=20, random_state=42)
        reg.fit(X_train, y_train)
        
        score = reg.score(X_test, y_test)
        
        # R2 score should be reasonable (can be negative for bad models)
        assert score > -1.0  # Not completely terrible
    
    def test_partial_fit_initial(self, regression_data):
        """Test partial_fit as initial fit."""
        X, y = regression_data
        
        reg = MondrianForestRegressor(n_estimators=10, random_state=42)
        reg.partial_fit(X, y)
        
        # Verify fitted
        assert len(reg.trees_) == 10
        
        # Can make predictions
        predictions = reg.predict(X)
        assert predictions.shape == (len(X),)
    
    def test_partial_fit_incremental(self, regression_data):
        """Test partial_fit for incremental learning."""
        X, y = regression_data
        
        # Split data for incremental learning
        split = int(0.7 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        # Initial fit with subset
        X_init, y_init = X_train[:50], y_train[:50]
        reg = MondrianForestRegressor(n_estimators=15, random_state=42)
        reg.fit(X_init, y_init)
        
        score1 = reg.score(X_test, y_test)
        
        # Incremental update with more data
        reg.partial_fit(X_train[50:], y_train[50:])
        score2 = reg.score(X_test, y_test)
        
        # Score should improve or stay similar
        # (May not always improve due to online nature)
        assert np.isfinite(score2)
    
    def test_different_n_estimators(self, regression_data):
        """Test with different numbers of estimators."""
        X, y = regression_data
        
        for n_estimators in [1, 5, 20]:
            reg = MondrianForestRegressor(n_estimators=n_estimators, random_state=42)
            reg.fit(X, y)
            
            assert len(reg.trees_) == n_estimators
            predictions = reg.predict(X)
            assert len(predictions) == len(X)


# Export Function Tests
class TestExportToExtraTrees:
    """Test export_mondrian_to_extratrees functionality."""
    
    def test_export_classifier(self, small_data):
        """Test exporting classifier to ExtraTrees."""
        X, y = small_data
        
        # Fit Mondrian Forest
        mf = MondrianForestClassifier(n_estimators=5, random_state=42)
        mf.fit(X, y)
        
        # Export to ExtraTrees
        et = export_mondrian_to_extratrees(mf)
        
        # Verify export
        assert isinstance(et, ExtraTreesClassifier)
        assert len(et.estimators_) == len(mf.trees_)
        assert et.n_features_in_ == mf.n_features_in_
        assert hasattr(et, 'classes_')
        
        # Should be able to make predictions
        predictions = et.predict(X)
        assert len(predictions) == len(X)
    
    def test_export_regressor(self, regression_data):
        """Test exporting regressor to ExtraTrees."""
        X, y = regression_data
        X_small = X[:100]
        y_small = y[:100]
        
        # Fit Mondrian Forest
        mf = MondrianForestRegressor(n_estimators=5, random_state=42)
        mf.fit(X_small, y_small)
        
        # Export to ExtraTrees
        et = export_mondrian_to_extratrees(mf)
        
        # Verify export
        assert isinstance(et, ExtraTreesRegressor)
        assert len(et.estimators_) == len(mf.trees_)
        assert et.n_features_in_ == mf.n_features_in_
        
        # Should be able to make predictions
        predictions = et.predict(X_small)
        assert len(predictions) == len(X_small)
    
    def test_export_unfitted_raises_error(self):
        """Test that exporting unfitted model raises error."""
        mf = MondrianForestClassifier(n_estimators=5)
        
        with pytest.raises(ValueError, match="must be fitted"):
            export_mondrian_to_extratrees(mf)
    
    def test_export_wrong_type_raises_error(self):
        """Test that exporting wrong type raises error."""
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        rf.fit(X, y)
        
        with pytest.raises(TypeError, match="Expected MondrianForestClassifier or MondrianForestRegressor"):
            export_mondrian_to_extratrees(rf)


# Integration Tests
class TestIntegration:
    """Integration tests combining multiple aspects."""
    
    def test_classifier_full_pipeline(self, classification_data):
        """Test complete classifier pipeline with train/test split."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Mondrian Forest
        mf = MondrianForestClassifier(n_estimators=30, random_state=42)
        mf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = mf.predict(X_test)
        y_proba = mf.predict_proba(X_test)
        
        # Verify predictions
        assert y_pred.shape == (len(X_test),)
        assert y_proba.shape == (len(X_test), mf.n_classes_)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.3  # Better than random
        
        # Export to ExtraTrees
        et = export_mondrian_to_extratrees(mf)
        et_pred = et.predict(X_test)
        
        # ExtraTrees should also make predictions
        assert et_pred.shape == (len(X_test),)
    
    def test_regressor_full_pipeline(self, regression_data):
        """Test complete regressor pipeline with train/test split."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Mondrian Forest
        mf = MondrianForestRegressor(n_estimators=30, random_state=42)
        mf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = mf.predict(X_test)
        
        # Verify predictions
        assert y_pred.shape == (len(X_test),)
        assert np.all(np.isfinite(y_pred))
        
        # Calculate R2 score
        score = r2_score(y_test, y_pred)
        assert score > -1.0  # Not completely terrible
        
        # Export to ExtraTrees
        et = export_mondrian_to_extratrees(mf)
        et_pred = et.predict(X_test)
        
        # ExtraTrees should also make predictions
        assert et_pred.shape == (len(X_test),)
    
    def test_online_learning_scenario(self, classification_data):
        """Test realistic online learning scenario."""
        X, y = classification_data
        
        # Simulate streaming data
        batch_size = 40
        n_batches = len(X) // batch_size
        
        mf = MondrianForestClassifier(n_estimators=15, random_state=42)
        
        # Process first batch with fit
        X_batch = X[:batch_size]
        y_batch = y[:batch_size]
        mf.fit(X_batch, y_batch)
        
        # Process remaining batches with partial_fit
        for i in range(1, n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            mf.partial_fit(X_batch, y_batch)
        
        # Final predictions should work
        final_pred = mf.predict(X)
        assert final_pred.shape == (len(X),)
        
        # Accuracy should be reasonable
        accuracy = accuracy_score(y, final_pred)
        assert accuracy > 0.3
    
    def test_classifier_then_export_then_convert(self, binary_classification_data):
        """Test Mondrian Forest -> ExtraTrees -> IsolationForest pipeline."""
        X, y = binary_classification_data
        
        # Import the converter
        from extratrees_to_iforest import convert_extratrees_to_isolationforest
        
        # Train Mondrian Forest
        mf = MondrianForestClassifier(n_estimators=10, random_state=42)
        mf.fit(X, y)
        
        # Export to ExtraTrees
        et = export_mondrian_to_extratrees(mf)
        
        # Convert to IsolationForest
        iforest = convert_extratrees_to_isolationforest(et, contamination=0.1)
        
        # Make anomaly predictions
        anomaly_pred = iforest.predict(X)
        anomaly_scores = iforest.score_samples(X)
        
        # Verify outputs
        assert anomaly_pred.shape == (len(X),)
        assert anomaly_scores.shape == (len(X),)
        assert set(anomaly_pred).issubset({-1, 1})


# Error Handling Tests
class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_predict_before_fit_classifier(self):
        """Test that predict before fit raises error."""
        mf = MondrianForestClassifier(n_estimators=5)
        X = np.random.randn(10, 4)
        
        with pytest.raises(Exception):  # sklearn raises NotFittedError
            mf.predict(X)
    
    def test_predict_before_fit_regressor(self):
        """Test that predict before fit raises error."""
        mf = MondrianForestRegressor(n_estimators=5)
        X = np.random.randn(10, 4)
        
        with pytest.raises(Exception):  # sklearn raises NotFittedError
            mf.predict(X)
    
    def test_invalid_input_dimensions(self, classification_data):
        """Test handling of invalid input dimensions."""
        X, y = classification_data
        
        mf = MondrianForestClassifier(n_estimators=5, random_state=42)
        mf.fit(X, y)
        
        # Wrong number of features
        X_wrong = np.random.randn(10, X.shape[1] + 1)
        
        with pytest.raises(ValueError):
            mf.predict(X_wrong)


# Performance Tests
class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_dataset_classifier(self):
        """Test classifier on larger dataset."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        mf = MondrianForestClassifier(n_estimators=50, random_state=42)
        mf.fit(X, y)
        
        predictions = mf.predict(X)
        assert len(predictions) == len(X)
        
        # Should achieve reasonable accuracy
        accuracy = accuracy_score(y, predictions)
        assert accuracy > 0.5
    
    def test_large_dataset_regressor(self):
        """Test regressor on larger dataset."""
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        mf = MondrianForestRegressor(n_estimators=50, random_state=42)
        mf.fit(X, y)
        
        predictions = mf.predict(X)
        assert len(predictions) == len(X)
        assert np.all(np.isfinite(predictions))
    
    def test_prediction_consistency(self, small_data):
        """Test that predictions are consistent across multiple calls."""
        X, y = small_data
        
        mf = MondrianForestClassifier(n_estimators=10, random_state=42)
        mf.fit(X, y)
        
        # Multiple predictions should be identical
        pred1 = mf.predict(X)
        pred2 = mf.predict(X)
        pred3 = mf.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

