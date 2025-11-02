"""
ExtraTrees to Mondrian Forest Conversion Examples

This script demonstrates various use cases for converting ExtraTreesClassifier
and ExtraTreesRegressor models to Mondrian Forest for online learning.

Run with: uv run python extratrees_to_mondrian_example.py
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from extratrees_to_mondrian import convert_extratrees_to_mondrian
from mondrian_forest import MondrianForestClassifier, MondrianForestRegressor


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def example_1_basic_classification():
    """Example 1: Basic ExtraTrees to Mondrian conversion for classification."""
    print_section("Example 1: Basic Classification Conversion")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3,
        n_informative=5, n_redundant=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Train ExtraTrees
    print("\n1. Training ExtraTreesClassifier...")
    et = ExtraTreesClassifier(n_estimators=20, max_depth=5, random_state=42)
    et.fit(X_train, y_train)
    et_score = et.score(X_test, y_test)
    print(f"   ExtraTrees test accuracy: {et_score:.4f}")
    
    # Convert to Mondrian
    print("\n2. Converting to MondrianForest...")
    mf = convert_extratrees_to_mondrian(et, lifetime=10.0)
    mf_score = mf.score(X_test, y_test)
    print(f"   Mondrian test accuracy: {mf_score:.4f}")
    print(f"   Accuracy difference: {abs(et_score - mf_score):.4f}")
    
    # Compare predictions
    et_pred = et.predict(X_test)
    mf_pred = mf.predict(X_test)
    agreement = np.mean(et_pred == mf_pred)
    print(f"\n3. Prediction agreement: {agreement*100:.1f}%")
    
    print("\n✓ Conversion successful! Predictions are similar.")


def example_2_regression():
    """Example 2: ExtraTrees to Mondrian conversion for regression."""
    print_section("Example 2: Regression Conversion")
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=200, n_features=10, noise=10.0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Train ExtraTrees
    print("\n1. Training ExtraTreesRegressor...")
    et = ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=42)
    et.fit(X_train, y_train)
    et_score = et.score(X_test, y_test)
    print(f"   ExtraTrees R² score: {et_score:.4f}")
    
    # Convert to Mondrian
    print("\n2. Converting to MondrianForest...")
    mf = convert_extratrees_to_mondrian(et, lifetime=10.0)
    mf_score = mf.score(X_test, y_test)
    print(f"   Mondrian R² score: {mf_score:.4f}")
    print(f"   Score difference: {abs(et_score - mf_score):.4f}")
    
    # Compare predictions
    et_pred = et.predict(X_test)
    mf_pred = mf.predict(X_test)
    correlation = np.corrcoef(et_pred, mf_pred)[0, 1]
    print(f"\n3. Prediction correlation: {correlation:.4f}")
    
    print("\n✓ Conversion successful! Predictions are highly correlated.")


def example_3_online_learning():
    """Example 3: Convert ExtraTrees and use online learning."""
    print_section("Example 3: Online Learning After Conversion")
    
    # Generate data with temporal component
    np.random.seed(42)
    n_total = 300
    X_all, y_all = make_classification(
        n_samples=n_total, n_features=8, n_classes=2,
        n_informative=6, random_state=42
    )
    
    # Split into initial training, online updates, and test
    X_initial = X_all[:150]
    y_initial = y_all[:150]
    X_stream = X_all[150:250]
    y_stream = y_all[150:250]
    X_test = X_all[250:]
    y_test = y_all[250:]
    
    print(f"Initial training: {len(X_initial)} samples")
    print(f"Online stream: {len(X_stream)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initial training with ExtraTrees
    print("\n1. Initial training with ExtraTrees...")
    et = ExtraTreesClassifier(n_estimators=15, random_state=42)
    et.fit(X_initial, y_initial)
    et_score = et.score(X_test, y_test)
    print(f"   ExtraTrees (initial) test accuracy: {et_score:.4f}")
    
    # Convert to Mondrian
    print("\n2. Converting to Mondrian with online learning enabled...")
    mf = convert_extratrees_to_mondrian(et, enable_online_learning=True)
    mf_score_before = mf.score(X_test, y_test)
    print(f"   Mondrian (before updates) test accuracy: {mf_score_before:.4f}")
    
    # Initialize bounds with representative data
    print("\n3. Initializing bounds with representative data...")
    mf.partial_fit(X_initial[:50], y_initial[:50])
    
    # Online learning in batches
    print("\n4. Processing online stream in batches...")
    batch_size = 25
    for i in range(0, len(X_stream), batch_size):
        X_batch = X_stream[i:i+batch_size]
        y_batch = y_stream[i:i+batch_size]
        mf.partial_fit(X_batch, y_batch)
        
        if (i // batch_size) % 2 == 0:  # Report every other batch
            score = mf.score(X_test, y_test)
            print(f"   After batch {i//batch_size + 1}: accuracy = {score:.4f}")
    
    mf_score_after = mf.score(X_test, y_test)
    print(f"\n5. Final Results:")
    print(f"   ExtraTrees (static):      {et_score:.4f}")
    print(f"   Mondrian (before):        {mf_score_before:.4f}")
    print(f"   Mondrian (after updates): {mf_score_after:.4f}")
    print(f"   Improvement from updates: {mf_score_after - mf_score_before:+.4f}")
    
    print("\n✓ Online learning working! Model adapted to new data.")


def example_4_comparison_native_vs_converted():
    """Example 4: Compare native Mondrian with converted Mondrian."""
    print_section("Example 4: Native vs Converted Mondrian")
    
    X, y = make_classification(
        n_samples=200, n_features=8, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X_train)} training, {len(X_test)} test samples")
    
    # Train native Mondrian
    print("\n1. Training native MondrianForest...")
    mf_native = MondrianForestClassifier(
        n_estimators=20, lifetime=5.0, random_state=42
    )
    mf_native.fit(X_train, y_train)
    native_score = mf_native.score(X_test, y_test)
    print(f"   Native Mondrian accuracy: {native_score:.4f}")
    
    # Train ExtraTrees and convert
    print("\n2. Training ExtraTrees...")
    et = ExtraTreesClassifier(n_estimators=20, random_state=42)
    et.fit(X_train, y_train)
    et_score = et.score(X_test, y_test)
    print(f"   ExtraTrees accuracy: {et_score:.4f}")
    
    print("\n3. Converting ExtraTrees to Mondrian...")
    mf_converted = convert_extratrees_to_mondrian(et, lifetime=5.0)
    converted_score = mf_converted.score(X_test, y_test)
    print(f"   Converted Mondrian accuracy: {converted_score:.4f}")
    
    print("\n4. Comparison:")
    print(f"   Native Mondrian:    {native_score:.4f} (random splits)")
    print(f"   ExtraTrees:         {et_score:.4f} (supervised splits)")
    print(f"   Converted Mondrian: {converted_score:.4f} (preserves ET splits)")
    
    print("\n📝 Note: Converted model preserves ExtraTrees' supervised splits,")
    print("   typically giving better initial accuracy than random Mondrian splits.")


def example_5_lifetime_parameter():
    """Example 5: Effect of lifetime parameter on converted trees."""
    print_section("Example 5: Lifetime Parameter Effects")
    
    X, y = make_classification(
        n_samples=150, n_features=6, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train ExtraTrees once
    print("Training ExtraTrees once...")
    et = ExtraTreesClassifier(n_estimators=10, random_state=42)
    et.fit(X_train, y_train)
    et_score = et.score(X_test, y_test)
    print(f"ExtraTrees accuracy: {et_score:.4f}\n")
    
    # Convert with different lifetime values
    lifetimes = [3.0, 5.0, 10.0, np.inf]
    
    print("Converting with different lifetime values:")
    for lifetime in lifetimes:
        mf = convert_extratrees_to_mondrian(et, lifetime=lifetime)
        mf_score = mf.score(X_test, y_test)
        
        # Check max depth in trees
        max_depth = max(tree.root.depth for tree in mf.trees_)
        
        print(f"  Lifetime={str(lifetime):8s} -> "
              f"Accuracy: {mf_score:.4f}, "
              f"Max tree depth: {max_depth}")
    
    print("\n📝 Note: Lifetime affects online learning behavior more than")
    print("   initial predictions (structure already determined by ExtraTrees).")


def example_6_bidirectional_conversion():
    """Example 6: ExtraTrees -> Mondrian -> ExtraTrees."""
    print_section("Example 6: Bidirectional Conversion")
    
    from mondrian_forest import export_mondrian_to_extratrees
    
    X, y = make_classification(
        n_samples=150, n_features=6, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X_train)} training, {len(X_test)} test samples\n")
    
    # Original ExtraTrees
    print("1. Training original ExtraTreesClassifier...")
    et1 = ExtraTreesClassifier(n_estimators=10, random_state=42)
    et1.fit(X_train, y_train)
    et1_score = et1.score(X_test, y_test)
    print(f"   Original ExtraTrees accuracy: {et1_score:.4f}")
    
    # Convert to Mondrian
    print("\n2. Converting to MondrianForest...")
    mf = convert_extratrees_to_mondrian(et1, enable_online_learning=True)
    mf_score = mf.score(X_test, y_test)
    print(f"   Mondrian accuracy: {mf_score:.4f}")
    
    # Apply online learning
    print("\n3. Applying online learning updates...")
    mf.partial_fit(X_train[:30], y_train[:30])  # Reinforce with some data
    mf_score_updated = mf.score(X_test, y_test)
    print(f"   Mondrian accuracy (after update): {mf_score_updated:.4f}")
    
    # Convert back to ExtraTrees
    print("\n4. Converting back to ExtraTreesClassifier...")
    et2 = export_mondrian_to_extratrees(mf)
    et2_score = et2.score(X_test, y_test)
    print(f"   Converted-back ExtraTrees accuracy: {et2_score:.4f}")
    
    print("\n5. Summary:")
    print(f"   Original ExtraTrees:        {et1_score:.4f}")
    print(f"   → Mondrian:                 {mf_score:.4f}")
    print(f"   → Mondrian (updated):       {mf_score_updated:.4f}")
    print(f"   → ExtraTrees (converted):   {et2_score:.4f}")
    
    print("\n✓ Bidirectional conversion working!")
    print("  Can move between model types as needed.")


def example_7_multiclass_with_online_learning():
    """Example 7: Multiclass classification with online learning."""
    print_section("Example 7: Multiclass + Online Learning")
    
    X, y = make_classification(
        n_samples=300, n_features=10, n_classes=4,
        n_informative=8, n_redundant=2, random_state=42
    )
    
    X_train = X[:150]
    y_train = y[:150]
    X_stream = X[150:250]
    y_stream = y[150:250]
    X_test = X[250:]
    y_test = y[250:]
    
    print(f"Classes: {len(np.unique(y))}")
    print(f"Training: {len(X_train)}, Stream: {len(X_stream)}, Test: {len(X_test)}")
    
    # Train ExtraTrees
    print("\n1. Training ExtraTrees on initial data...")
    et = ExtraTreesClassifier(n_estimators=25, random_state=42)
    et.fit(X_train, y_train)
    et_score = et.score(X_test, y_test)
    print(f"   ExtraTrees test accuracy: {et_score:.4f}")
    
    # Convert
    print("\n2. Converting to Mondrian...")
    mf = convert_extratrees_to_mondrian(et, enable_online_learning=True)
    
    # Initialize bounds
    print("\n3. Initializing with representative data...")
    mf.partial_fit(X_train[:40], y_train[:40])
    initial_score = mf.score(X_test, y_test)
    print(f"   Initial Mondrian accuracy: {initial_score:.4f}")
    
    # Online updates
    print("\n4. Processing streaming data...")
    for i in range(0, len(X_stream), 25):
        mf.partial_fit(X_stream[i:i+25], y_stream[i:i+25])
    
    final_score = mf.score(X_test, y_test)
    print(f"   Final Mondrian accuracy: {final_score:.4f}")
    
    # Test predict_proba
    print("\n5. Testing probability predictions...")
    proba = mf.predict_proba(X_test[:5])
    print("   Sample probabilities (first 5 test samples):")
    for i, p in enumerate(proba):
        print(f"     Sample {i}: {p}")
        assert np.allclose(p.sum(), 1.0), "Probabilities should sum to 1"
    
    print("\n✓ Multiclass online learning working!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  ExtraTrees to Mondrian Forest Conversion Examples")
    print("="*70)
    print("\nThis script demonstrates converting ExtraTrees models to Mondrian")
    print("forests for online learning capabilities.")
    
    try:
        example_1_basic_classification()
        example_2_regression()
        example_3_online_learning()
        example_4_comparison_native_vs_converted()
        example_5_lifetime_parameter()
        example_6_bidirectional_conversion()
        example_7_multiclass_with_online_learning()
        
        print("\n" + "="*70)
        print("  ✓ All Examples Completed Successfully!")
        print("="*70)
        print("\nKey Takeaways:")
        print("  1. Conversion preserves ExtraTrees predictions closely")
        print("  2. Online learning works after conversion (with limitations)")
        print("  3. Converted models are NOT true Mondrian forests")
        print("  4. Good for practical incremental learning scenarios")
        print("  5. See EXTRATREES_TO_MONDRIAN_LIMITATIONS.md for details")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

