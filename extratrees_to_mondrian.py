"""
ExtraTrees to Mondrian Forest Converter

This module provides functionality to convert trained ExtraTreesClassifier or
ExtraTreesRegressor models into MondrianForest objects for online learning.

Public API:
    - convert_extratrees_to_mondrian: Main conversion function

IMPORTANT LIMITATIONS:
    This conversion is a structural approximation. The resulting Mondrian trees
    will not have true Mondrian process properties (random space partitioning
    with exponentially distributed time parameters). Instead, they preserve the
    supervised learning splits from the ExtraTrees model.

"""

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.utils import check_array
from mondrian_forest import (
    MondrianForestClassifier,
    MondrianForestRegressor,
    MondrianTree,
    MondrianNode
)
import numpy as np


def convert_extratrees_to_mondrian(
    extratrees_model,
    lifetime=np.inf,
    enable_online_learning=True
):
    """
    Convert an ExtraTreesClassifier or ExtraTreesRegressor to a Mondrian Forest.
    
    This function creates a new Mondrian Forest and transfers the tree structures
    from the ExtraTrees model. The conversion preserves the split features and
    thresholds, but synthesizes Mondrian-specific attributes (tau, delta, bounds).
    
    IMPORTANT: The resulting Mondrian trees are structural approximations and
    do not possess true Mondrian process properties. The splits come from
    supervised learning (ExtraTrees) rather than random Mondrian partitioning.
    
    The conversion process:
    1. Extracts parameters from the ExtraTrees model
    2. Creates a new Mondrian Forest with compatible parameters
    3. Converts each sklearn tree to a MondrianTree structure
    4. Synthesizes Mondrian-specific attributes (tau, delta)
    5. Preserves split features, thresholds, and predictions
    6. Optionally enables bounds tracking for online learning
    
    Parameters
    ----------
    extratrees_model : ExtraTreesClassifier or ExtraTreesRegressor
        A fitted ExtraTreesClassifier or ExtraTreesRegressor model to convert.
        Must be already trained (have estimators_ attribute).
        
    lifetime : float, default=np.inf
        Maximum lifetime (tau) for the Mondrian trees. Controls tree depth
        in online learning updates. Set to np.inf for unlimited depth.
        
    enable_online_learning : bool, default=True
        If True, initializes bounding boxes to None, allowing them to be
        computed when partial_fit is first called. If False, attempts to
        estimate bounds from thresholds (less accurate).
    
    Returns
    -------
    mondrian_forest : MondrianForestClassifier or MondrianForestRegressor
        A new Mondrian Forest object with tree structures converted from the
        input model. Supports:
        - predict(X): Make predictions using converted trees
        - partial_fit(X, y): Online learning (with limitations)
        - predict_proba(X): Class probabilities (classifier only)
    
    Raises
    ------
    TypeError
        If extratrees_model is not an ExtraTreesClassifier or ExtraTreesRegressor.
    ValueError
        If the ExtraTrees model has not been fitted (no estimators_ attribute).
    
    Limitations and Assumptions
    ---------------------------
    1. **Not True Mondrian Trees**: The converted trees preserve ExtraTrees splits
       rather than random Mondrian partitions. They won't have the theoretical
       properties of true Mondrian forests.
       
    2. **Synthesized Time Parameters**: tau and delta are approximated based on
       tree depth (tau ≈ depth), not from a true Mondrian process.
       
    3. **Bounding Boxes**: Original bounding boxes are lost. With
       enable_online_learning=True, they're set to None and computed on first
       partial_fit. With False, rough estimates are made from thresholds.
       
    4. **Online Learning Limitations**: While partial_fit will work, the online
       updates won't follow true Mondrian dynamics. New splits will extend the
       tree, but not according to the Mondrian process.
       
    5. **Prediction Behavior**: Initial predictions match ExtraTrees closely,
       but may diverge after online updates.
       
    6. **Bootstrap Information Lost**: ExtraTrees bootstrap indices are not
       preserved, affecting exact prediction reproducibility.
       
    7. **Class Encoding**: For classifiers, ensure the same class encoding is
       used if you call partial_fit after conversion.
    
    Examples
    --------
    >>> from sklearn.ensemble import ExtraTreesClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    >>> 
    >>> # Train ExtraTrees model
    >>> et = ExtraTreesClassifier(n_estimators=10, random_state=42)
    >>> et.fit(X, y)
    >>> 
    >>> # Convert to Mondrian Forest
    >>> mf = convert_extratrees_to_mondrian(et, lifetime=10.0)
    >>> 
    >>> # Use for prediction
    >>> predictions = mf.predict(X)
    >>> 
    >>> # Use for online learning (with limitations)
    >>> mf.partial_fit(X[:10], y[:10])
    
    Using ExtraTreesRegressor:
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    >>> et_reg = ExtraTreesRegressor(n_estimators=10, random_state=42)
    >>> et_reg.fit(X, y)
    >>> mf_reg = convert_extratrees_to_mondrian(et_reg)
    >>> predictions = mf_reg.predict(X)
    
    Disabling online learning preparation:
    >>> # If you don't plan to use partial_fit, disable bound initialization
    >>> mf = convert_extratrees_to_mondrian(et, enable_online_learning=False)
    >>> # partial_fit will not work reliably in this mode
    
    Notes
    -----
    The converted Mondrian Forest inherits tree structures from ExtraTrees,
    which were trained using supervised learning criteria (gini, entropy, MSE).
    This fundamentally differs from native Mondrian forests that use random
    splits via the Mondrian process. Use this converter when you:
    
    1. Have a trained ExtraTrees model and want to add online learning
    2. Want to experiment with incremental updates to existing models
    3. Need to convert between model types for compatibility
    
    Do NOT use this converter if you need:
    - True Mondrian process properties for theoretical guarantees
    - Exact reproducibility of Mondrian forest behavior
    - Pure random partitioning for anomaly detection
    
    See Also
    --------
    mondrian_forest.MondrianForestClassifier : Target classifier class
    mondrian_forest.MondrianForestRegressor : Target regressor class
    sklearn.ensemble.ExtraTreesClassifier : Source classifier model
    sklearn.ensemble.ExtraTreesRegressor : Source regressor model
    """
    # Validate input type
    is_classifier = isinstance(extratrees_model, ExtraTreesClassifier)
    is_regressor = isinstance(extratrees_model, ExtraTreesRegressor)
    
    if not (is_classifier or is_regressor):
        raise TypeError(
            f"Expected ExtraTreesClassifier or ExtraTreesRegressor, "
            f"got {type(extratrees_model).__name__}"
        )
    
    # Validate that model is fitted
    if not hasattr(extratrees_model, 'estimators_'):
        raise ValueError(
            "The ExtraTrees model must be fitted before conversion. "
            "Call fit() on the model first."
        )
    
    # Extract parameters
    n_estimators = len(extratrees_model.estimators_)
    random_state = extratrees_model.random_state
    n_features = extratrees_model.n_features_in_
    
    # Create Mondrian Forest
    if is_classifier:
        mondrian_forest = MondrianForestClassifier(
            n_estimators=n_estimators,
            lifetime=lifetime,
            random_state=random_state
        )
        mondrian_forest.classes_ = extratrees_model.classes_
        mondrian_forest.n_classes_ = extratrees_model.n_classes_
        mondrian_forest.class_to_idx_ = {
            c: i for i, c in enumerate(mondrian_forest.classes_)
        }
    else:
        mondrian_forest = MondrianForestRegressor(
            n_estimators=n_estimators,
            lifetime=lifetime,
            random_state=random_state
        )
    
    mondrian_forest.n_features_in_ = n_features
    
    # Convert each tree
    mondrian_forest.trees_ = []
    for i, sklearn_tree in enumerate(extratrees_model.estimators_):
        mondrian_tree = MondrianTree(
            lifetime=lifetime,
            random_state=random_state
        )
        mondrian_tree.is_classifier = is_classifier
        if is_classifier:
            mondrian_tree.n_classes = mondrian_forest.n_classes_
        
        # Convert sklearn tree to Mondrian tree
        mondrian_tree.root = _convert_sklearn_tree_to_mondrian_node(
            sklearn_tree.tree_,
            node_id=0,
            depth=0,
            tau=0.0,
            is_classifier=is_classifier,
            n_classes=mondrian_forest.n_classes_ if is_classifier else None,
            n_features=n_features,
            enable_online_learning=enable_online_learning
        )
        
        mondrian_forest.trees_.append(mondrian_tree)
    
    return mondrian_forest


def _convert_sklearn_tree_to_mondrian_node(
    sklearn_tree,
    node_id,
    depth,
    tau,
    is_classifier,
    n_classes,
    n_features,
    enable_online_learning
):
    """
    Recursively convert an sklearn tree structure to MondrianNode structure.
    
    Parameters
    ----------
    sklearn_tree : sklearn.tree._tree.Tree
        The sklearn tree structure
    node_id : int
        Current node ID in the sklearn tree
    depth : int
        Current depth in the tree
    tau : float
        Current tau (time parameter) for this node
    is_classifier : bool
        Whether this is a classification tree
    n_classes : int or None
        Number of classes (for classification)
    n_features : int
        Number of features in the dataset
    enable_online_learning : bool
        Whether to enable online learning (affects bound initialization)
    
    Returns
    -------
    node : MondrianNode
        Converted Mondrian node
    """
    # Create new Mondrian node
    node = MondrianNode(tau=tau)
    node.depth = depth
    node.samples_seen = sklearn_tree.n_node_samples[node_id]
    
    # Extract feature and threshold
    feature = sklearn_tree.feature[node_id]
    threshold = sklearn_tree.threshold[node_id]
    
    # Check if leaf node
    # In sklearn, leaf nodes have feature == -2
    if feature == -2:
        node.is_leaf = True
        node.feature = None
        node.threshold = None
        
        # Extract prediction from value
        value = sklearn_tree.value[node_id]
        
        if is_classifier:
            # For classification, value is class counts
            # sklearn stores as [[count0, count1, ...]]
            node.prediction = value[0].astype(float)
        else:
            # For regression, value is the mean
            # sklearn stores as [[mean]]
            node.prediction = float(value[0, 0])
        
        # Initialize bounds
        if enable_online_learning:
            # Set to None - will be computed on first partial_fit
            node.lower_bounds = None
            node.upper_bounds = None
        else:
            # Create placeholder bounds (won't be accurate)
            node.lower_bounds = np.zeros(n_features)
            node.upper_bounds = np.zeros(n_features)
    
    else:
        # Internal node
        node.is_leaf = False
        node.feature = int(feature)
        node.threshold = float(threshold)
        
        # Synthesize delta (time increment)
        # Use a simple approximation: delta = 1.0 per level
        # This doesn't follow true Mondrian process but provides structure
        delta = 1.0
        node.delta = delta
        
        # Extract prediction (internal nodes can also make predictions)
        value = sklearn_tree.value[node_id]
        if is_classifier:
            node.prediction = value[0].astype(float)
        else:
            node.prediction = float(value[0, 0])
        
        # Initialize bounds
        if enable_online_learning:
            node.lower_bounds = None
            node.upper_bounds = None
        else:
            # Rough estimate based on threshold
            # This is very approximate and won't be accurate
            node.lower_bounds = np.full(n_features, -1000.0)
            node.upper_bounds = np.full(n_features, 1000.0)
            node.lower_bounds[feature] = threshold - 10.0
            node.upper_bounds[feature] = threshold + 10.0
        
        # Recursively convert children
        left_child_id = sklearn_tree.children_left[node_id]
        right_child_id = sklearn_tree.children_right[node_id]
        
        if left_child_id != -1:
            node.left = _convert_sklearn_tree_to_mondrian_node(
                sklearn_tree,
                left_child_id,
                depth + 1,
                tau + delta,
                is_classifier,
                n_classes,
                n_features,
                enable_online_learning
            )
        
        if right_child_id != -1:
            node.right = _convert_sklearn_tree_to_mondrian_node(
                sklearn_tree,
                right_child_id,
                depth + 1,
                tau + delta,
                is_classifier,
                n_classes,
                n_features,
                enable_online_learning
            )
    
    return node


# Public API exports
__all__ = ['convert_extratrees_to_mondrian']

