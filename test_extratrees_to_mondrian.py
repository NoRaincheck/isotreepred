"""
Tests for ExtraTrees to Mondrian Forest Converter

This test suite verifies the conversion functionality from ExtraTreesClassifier
and ExtraTreesRegressor to MondrianForest models.
"""

import pytest
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from extratrees_to_mondrian import convert_extratrees_to_mondrian
from mondrian_forest import MondrianForestClassifier, MondrianForestRegressor


class TestBasicConversion:
    """Test basic conversion functionality."""
    
    def test_convert_classifier(self):
        """Test converting ExtraTreesClassifier to MondrianForestClassifier."""
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=2, random_state=42
        )
        
        # Train ExtraTrees
        et = ExtraTreesClassifier(n_estimators=5, random_state=42)
        et.fit(X, y)
        
        # Convert to Mondrian
        mf = convert_extratrees_to_mondrian(et)
        
        # Check type
        assert isinstance(mf, MondrianForestClassifier)
        
        # Check attributes
        assert mf.n_estimators == 5
        assert mf.n_features_in_ == 4
        assert len(mf.trees_) == 5
        assert mf.n_classes_ == 2
        assert np.array_equal(mf.classes_, et.classes_)
    
    def test_convert_regressor(self):
        """Test converting ExtraTreesRegressor to MondrianForestRegressor."""
        X, y = make_regression(
            n_samples=100, n_features=4, random_state=42
        )
        
        # Train ExtraTrees
        et = ExtraTreesRegressor(n_estimators=5, random_state=42)
        et.fit(X, y)
        
        # Convert to Mondrian
        mf = convert_extratrees_to_mondrian(et)
        
        # Check type
        assert isinstance(mf, MondrianForestRegressor)
        
        # Check attributes
        assert mf.n_estimators == 5
        assert mf.n_features_in_ == 4
        assert len(mf.trees_) == 5
    
    def test_convert_with_lifetime(self):
        """Test conversion with custom lifetime parameter."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=3, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et, lifetime=5.0)
        
        assert mf.lifetime == 5.0
        assert all(tree.lifetime == 5.0 for tree in mf.trees_)
    
    def test_tree_structure_preserved(self):
        """Test that tree structure is preserved after conversion."""
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=1, max_depth=3, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        # Both should have trees
        assert len(et.estimators_) == 1
        assert len(mf.trees_) == 1
        
        # Check that root exists
        assert mf.trees_[0].root is not None
        
        # Check that tree has depth
        assert mf.trees_[0].root.depth == 0


class TestPredictions:
    """Test prediction functionality after conversion."""
    
    def test_classifier_predictions_similar(self):
        """Test that predictions are similar after conversion."""
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train ExtraTrees
        et = ExtraTreesClassifier(n_estimators=10, random_state=42)
        et.fit(X_train, y_train)
        et_pred = et.predict(X_test)
        
        # Convert and predict
        mf = convert_extratrees_to_mondrian(et)
        mf_pred = mf.predict(X_test)
        
        # Predictions should be similar (may not be identical due to averaging)
        # Check that at least 70% match
        accuracy = np.mean(et_pred == mf_pred)
        assert accuracy > 0.7, f"Only {accuracy*100:.1f}% predictions match"
    
    def test_regressor_predictions_similar(self):
        """Test that regressor predictions are similar after conversion."""
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train ExtraTrees
        et = ExtraTreesRegressor(n_estimators=10, random_state=42)
        et.fit(X_train, y_train)
        et_pred = et.predict(X_test)
        
        # Convert and predict
        mf = convert_extratrees_to_mondrian(et)
        mf_pred = mf.predict(X_test)
        
        # Predictions should be correlated
        correlation = np.corrcoef(et_pred, mf_pred)[0, 1]
        assert correlation > 0.8, f"Correlation only {correlation:.2f}"
    
    def test_predict_proba_works(self):
        """Test that predict_proba works after conversion."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=5, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        # Should not raise error
        proba = mf.predict_proba(X)
        
        # Check shape
        assert proba.shape == (100, 2)
        
        # Check probabilities sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_multiclass_classification(self):
        """Test conversion with multiclass classification."""
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=3,
            n_informative=3, n_redundant=0, random_state=42
        )
        
        et = ExtraTreesClassifier(n_estimators=5, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        assert mf.n_classes_ == 3
        
        predictions = mf.predict(X)
        assert len(np.unique(predictions)) <= 3
        
        proba = mf.predict_proba(X)
        assert proba.shape == (100, 3)


class TestOnlineLearning:
    """Test online learning functionality after conversion."""
    
    def test_partial_fit_works(self):
        """Test that partial_fit works after conversion."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=3, random_state=42)
        et.fit(X[:50], y[:50])
        
        mf = convert_extratrees_to_mondrian(et, enable_online_learning=True)
        
        # Should not raise error
        mf.partial_fit(X[50:60], y[50:60])
        
        # Should still be able to predict
        predictions = mf.predict(X)
        assert len(predictions) == 100
    
    def test_partial_fit_regressor(self):
        """Test partial_fit for regressor after conversion."""
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
        
        et = ExtraTreesRegressor(n_estimators=3, random_state=42)
        et.fit(X[:50], y[:50])
        
        mf = convert_extratrees_to_mondrian(et, enable_online_learning=True)
        
        # Should not raise error
        mf.partial_fit(X[50:60], y[50:60])
        
        # Should still be able to predict
        predictions = mf.predict(X)
        assert len(predictions) == 100
    
    def test_online_learning_disabled(self):
        """Test conversion with online learning disabled."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=3, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et, enable_online_learning=False)
        
        # Should still predict
        predictions = mf.predict(X)
        assert len(predictions) == 100


class TestErrorHandling:
    """Test error handling in conversion."""
    
    def test_unfitted_model_raises_error(self):
        """Test that converting unfitted model raises error."""
        et = ExtraTreesClassifier(n_estimators=5)
        
        with pytest.raises(ValueError, match="must be fitted"):
            convert_extratrees_to_mondrian(et)
    
    def test_wrong_type_raises_error(self):
        """Test that wrong model type raises error."""
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        rf.fit(X, y)
        
        with pytest.raises(TypeError, match="Expected ExtraTreesClassifier"):
            convert_extratrees_to_mondrian(rf)
    
    def test_none_input_raises_error(self):
        """Test that None input raises error."""
        with pytest.raises(TypeError):
            convert_extratrees_to_mondrian(None)


class TestNodeStructure:
    """Test the structure of converted nodes."""
    
    def test_leaf_nodes_have_predictions(self):
        """Test that leaf nodes have predictions."""
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=1, max_depth=2, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        # Traverse tree and check leaf nodes
        def check_leaves(node):
            if node.is_leaf:
                assert node.prediction is not None
                assert node.feature is None
                assert node.threshold is None
            else:
                assert node.feature is not None
                assert node.threshold is not None
                if node.left:
                    check_leaves(node.left)
                if node.right:
                    check_leaves(node.right)
        
        check_leaves(mf.trees_[0].root)
    
    def test_tau_increases_with_depth(self):
        """Test that tau increases as we go deeper in tree."""
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=1, max_depth=3, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        root = mf.trees_[0].root
        assert root.tau == 0.0
        
        # Check children have higher tau
        if root.left:
            assert root.left.tau > root.tau
        if root.right:
            assert root.right.tau > root.tau
    
    def test_samples_seen_preserved(self):
        """Test that samples_seen is preserved from sklearn."""
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=1, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        # Root should have seen all samples
        root = mf.trees_[0].root
        assert root.samples_seen == 50


class TestIntegration:
    """Integration tests with full pipelines."""
    
    def test_full_classification_pipeline(self):
        """Test full pipeline: ExtraTrees -> Mondrian -> Predict."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=3,
            n_informative=5, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train ExtraTrees
        et = ExtraTreesClassifier(n_estimators=20, random_state=42)
        et.fit(X_train, y_train)
        et_score = et.score(X_test, y_test)
        
        # Convert to Mondrian
        mf = convert_extratrees_to_mondrian(et)
        mf_score = mf.score(X_test, y_test)
        
        # Scores should be reasonably close
        assert abs(et_score - mf_score) < 0.2, \
            f"Scores differ too much: ET={et_score:.3f}, MF={mf_score:.3f}"
    
    def test_full_regression_pipeline(self):
        """Test full pipeline: ExtraTrees -> Mondrian -> Predict."""
        X, y = make_regression(
            n_samples=200, n_features=10, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train ExtraTrees
        et = ExtraTreesRegressor(n_estimators=20, random_state=42)
        et.fit(X_train, y_train)
        et_score = et.score(X_test, y_test)
        
        # Convert to Mondrian
        mf = convert_extratrees_to_mondrian(et)
        mf_score = mf.score(X_test, y_test)
        
        # Scores should be reasonably close
        assert abs(et_score - mf_score) < 0.2, \
            f"Scores differ too much: ET={et_score:.3f}, MF={mf_score:.3f}"
    
    def test_conversion_and_online_learning(self):
        """Test converting and then using online learning."""
        X, y = make_classification(n_samples=150, n_features=5, random_state=42)
        
        # Initial training with ExtraTrees
        et = ExtraTreesClassifier(n_estimators=10, random_state=42)
        et.fit(X[:100], y[:100])
        
        # Convert to Mondrian
        mf = convert_extratrees_to_mondrian(et, enable_online_learning=True)
        
        # Get initial predictions
        pred_before = mf.predict(X[100:])
        
        # Online update
        mf.partial_fit(X[100:125], y[100:125])
        
        # Predictions should still work
        pred_after = mf.predict(X[100:])
        
        assert len(pred_after) == 50
        # Predictions may change after online learning
    
    def test_bidirectional_conversion(self):
        """Test converting ExtraTrees->Mondrian->ExtraTrees."""
        from mondrian_forest import export_mondrian_to_extratrees
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        # Original ExtraTrees
        et1 = ExtraTreesClassifier(n_estimators=5, random_state=42)
        et1.fit(X, y)
        
        # Convert to Mondrian
        mf = convert_extratrees_to_mondrian(et1)
        
        # Convert back to ExtraTrees
        et2 = export_mondrian_to_extratrees(mf)
        
        # Should be able to predict
        pred1 = et1.predict(X)
        pred2 = et2.predict(X)
        
        # Both should predict something
        assert len(pred1) == len(pred2) == 100


class TestLargeDatasets:
    """Test conversion with larger datasets."""
    
    def test_large_dataset_classification(self):
        """Test conversion with larger dataset."""
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=4,
            n_informative=10, n_redundant=5, random_state=42
        )
        
        et = ExtraTreesClassifier(n_estimators=15, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        # Should complete without error
        predictions = mf.predict(X)
        assert len(predictions) == 1000
    
    def test_many_trees(self):
        """Test conversion with many trees."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        et = ExtraTreesClassifier(n_estimators=50, random_state=42)
        et.fit(X, y)
        
        mf = convert_extratrees_to_mondrian(et)
        
        assert len(mf.trees_) == 50
        
        predictions = mf.predict(X)
        assert len(predictions) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

