"""
Example usage of Mondrian Forest with online learning and export capabilities.

Run with: uv run python mondrian_example.py
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from mondrian_forest import (
    MondrianForestClassifier,
    MondrianForestRegressor,
    export_mondrian_to_extratrees
)
from extratrees_to_iforest import convert_extratrees_to_isolationforest


def example_classifier_basic():
    """Basic classifier example."""
    print("\n" + "="*60)
    print("Example 1: Basic Mondrian Forest Classifier")
    print("="*60)
    
    # Generate data
    X, y = make_classification(
        n_samples=500, 
        n_features=10, 
        n_informative=5,
        n_classes=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Mondrian Forest
    mf = MondrianForestClassifier(n_estimators=30, random_state=42)
    mf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = mf.predict(X_test)
    y_proba = mf.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of trees: {len(mf.trees_)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Probability shape: {y_proba.shape}")


def example_classifier_online_learning():
    """Online learning example with partial_fit."""
    print("\n" + "="*60)
    print("Example 2: Online Learning with partial_fit")
    print("="*60)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    
    # Split into batches for streaming
    batch_size = 100
    n_batches = len(X) // batch_size
    
    # Initialize with first batch
    mf = MondrianForestClassifier(n_estimators=20, random_state=42)
    X_batch = X[:batch_size]
    y_batch = y[:batch_size]
    mf.fit(X_batch, y_batch)
    
    print(f"Initial training: {batch_size} samples")
    initial_accuracy = accuracy_score(y[:batch_size], mf.predict(X[:batch_size]))
    print(f"Initial accuracy: {initial_accuracy:.3f}")
    
    # Process remaining batches with partial_fit
    for i in range(1, n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        mf.partial_fit(X_batch, y_batch)
        
        if (i + 1) % 3 == 0:
            current_accuracy = accuracy_score(
                y[:end_idx], 
                mf.predict(X[:end_idx])
            )
            print(f"After batch {i+1} ({end_idx} samples): accuracy = {current_accuracy:.3f}")
    
    # Final evaluation
    final_accuracy = accuracy_score(y, mf.predict(X))
    print(f"\nFinal accuracy (all {len(X)} samples): {final_accuracy:.3f}")


def example_regressor():
    """Regression example."""
    print("\n" + "="*60)
    print("Example 3: Mondrian Forest Regressor")
    print("="*60)
    
    # Generate data
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Mondrian Forest
    mf = MondrianForestRegressor(n_estimators=30, random_state=42)
    mf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = mf.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of trees: {len(mf.trees_)}")
    print(f"R² score: {r2:.3f}")
    print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"True value range: [{y_test.min():.2f}, {y_test.max():.2f}]")


def example_export_to_extratrees():
    """Export to ExtraTrees example."""
    print("\n" + "="*60)
    print("Example 4: Export Mondrian Forest to ExtraTrees")
    print("="*60)
    
    # Generate data
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=4,
        random_state=42
    )
    
    # Train Mondrian Forest
    mf = MondrianForestClassifier(n_estimators=15, random_state=42)
    mf.fit(X, y)
    
    # Export to ExtraTrees
    et = export_mondrian_to_extratrees(mf)
    
    # Compare predictions
    mf_pred = mf.predict(X)
    et_pred = et.predict(X)
    
    mf_accuracy = accuracy_score(y, mf_pred)
    et_accuracy = accuracy_score(y, et_pred)
    
    print(f"Mondrian Forest accuracy: {mf_accuracy:.3f}")
    print(f"ExtraTrees accuracy: {et_accuracy:.3f}")
    print(f"Number of estimators: {len(et.estimators_)}")
    print(f"ExtraTrees type: {type(et).__name__}")


def example_full_pipeline():
    """Complete pipeline: Mondrian Forest -> ExtraTrees -> IsolationForest."""
    print("\n" + "="*60)
    print("Example 5: Full Pipeline (MF -> ET -> IF)")
    print("="*60)
    
    # Generate data
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        random_state=42
    )
    
    # Step 1: Train Mondrian Forest
    print("\n1. Training Mondrian Forest...")
    mf = MondrianForestClassifier(n_estimators=20, random_state=42)
    mf.fit(X, y)
    mf_accuracy = accuracy_score(y, mf.predict(X))
    print(f"   Mondrian Forest accuracy: {mf_accuracy:.3f}")
    
    # Step 2: Export to ExtraTrees
    print("\n2. Exporting to ExtraTrees...")
    et = export_mondrian_to_extratrees(mf)
    et_accuracy = accuracy_score(y, et.predict(X))
    print(f"   ExtraTrees accuracy: {et_accuracy:.3f}")
    
    # Step 3: Convert to IsolationForest for anomaly detection
    print("\n3. Converting to IsolationForest...")
    iforest = convert_extratrees_to_isolationforest(et, contamination=0.1)
    
    # Detect anomalies
    anomaly_pred = iforest.predict(X)
    anomaly_scores = iforest.score_samples(X)
    
    n_outliers = np.sum(anomaly_pred == -1)
    n_inliers = np.sum(anomaly_pred == 1)
    
    print(f"   Outliers detected: {n_outliers} ({n_outliers/len(X)*100:.1f}%)")
    print(f"   Inliers: {n_inliers} ({n_inliers/len(X)*100:.1f}%)")
    print(f"   Anomaly score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
    
    print("\n✓ Complete pipeline executed successfully!")


def example_lifetime_parameter():
    """Demonstrate the lifetime parameter effect."""
    print("\n" + "="*60)
    print("Example 6: Lifetime Parameter Effect")
    print("="*60)
    
    # Generate data
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        random_state=42
    )
    
    # Short lifetime (shallower trees)
    mf_short = MondrianForestClassifier(
        n_estimators=10, 
        lifetime=1.0, 
        random_state=42
    )
    mf_short.fit(X, y)
    
    # Long lifetime (deeper trees)
    mf_long = MondrianForestClassifier(
        n_estimators=10,
        lifetime=100.0,
        random_state=42
    )
    mf_long.fit(X, y)
    
    # Infinite lifetime (no constraint)
    mf_inf = MondrianForestClassifier(
        n_estimators=10,
        lifetime=np.inf,
        random_state=42
    )
    mf_inf.fit(X, y)
    
    print(f"Short lifetime (1.0):")
    print(f"  Accuracy: {accuracy_score(y, mf_short.predict(X)):.3f}")
    
    print(f"\nLong lifetime (100.0):")
    print(f"  Accuracy: {accuracy_score(y, mf_long.predict(X)):.3f}")
    
    print(f"\nInfinite lifetime (np.inf):")
    print(f"  Accuracy: {accuracy_score(y, mf_inf.predict(X)):.3f}")
    
    print("\nNote: Lifetime controls tree depth - shorter lifetime = shallower trees")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MONDRIAN FOREST EXAMPLES")
    print("="*60)
    
    example_classifier_basic()
    example_classifier_online_learning()
    example_regressor()
    example_export_to_extratrees()
    example_full_pipeline()
    example_lifetime_parameter()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

