"""
ExtraTrees to IsolationForest Converter

This module provides functionality to convert trained ExtraTreesClassifier or
ExtraTreesRegressor models into IsolationForest objects for anomaly detection.

Public API:
    - convert_extratrees_to_isolationforest: Main conversion function

"""

from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.utils import check_array
import numpy as np
import copy


def convert_extratrees_to_isolationforest(
    extratrees_model, 
    contamination='auto', 
    offset=None
):
    """
    Convert an ExtraTreesClassifier or ExtraTreesRegressor to an IsolationForest object.
    
    This function creates a new IsolationForest instance and transfers the tree structures
    and relevant parameters from the ExtraTrees model to make it behave like an IsolationForest.
    
    The conversion process:
    1. Extracts parameters from the ExtraTrees model
    2. Creates a new IsolationForest with compatible parameters
    3. Deep copies all tree estimators
    4. Pre-computes path lengths required for anomaly scoring
    5. Sets all internal attributes required by IsolationForest
    
    Parameters
    ----------
    extratrees_model : ExtraTreesClassifier or ExtraTreesRegressor
        A fitted ExtraTreesClassifier or ExtraTreesRegressor model to convert.
        Must be already trained (have estimators_ attribute).
        
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
        - If 'auto', the threshold is determined automatically
        - If float, should be in the range (0, 0.5]
        
    offset : float, optional
        Offset used to define the decision function. If None, it will be computed
        based on the contamination parameter (default: -0.5 for 'auto').
    
    Returns
    -------
    iforest : IsolationForest
        A new IsolationForest object with tree structures copied from the input model.
        Supports all standard IsolationForest methods:
        - predict(X): Returns -1 for outliers, 1 for inliers
        - score_samples(X): Returns anomaly scores (higher = more normal)
        - decision_function(X): Returns decision function values
    
    Raises
    ------
    TypeError
        If extratrees_model is not an ExtraTreesClassifier or ExtraTreesRegressor.
    ValueError
        If the ExtraTrees model has not been fitted (no estimators_ attribute).
    
    Limitations
    -----------
    1. The ExtraTrees model must be fitted before conversion
    2. The conversion creates deep copies of trees (memory intensive for large models)
    3. The anomaly scores reflect the tree structure from supervised/regression learning,
       not pure isolation-based anomaly detection
    4. ONNX export only works when max_features=None (all features) due to skl2onnx
       converter limitations
    5. The resulting IsolationForest uses ExtraTrees splitting criteria rather than
       IsolationForest's random splits
    
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
    >>> # Convert to IsolationForest
    >>> iforest = convert_extratrees_to_isolationforest(et, contamination=0.1)
    >>> 
    >>> # Use for anomaly detection
    >>> predictions = iforest.predict(X)
    >>> scores = iforest.score_samples(X)
    >>> print(f"Found {sum(predictions == -1)} outliers")
    
    Using ExtraTreesRegressor:
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> et_reg = ExtraTreesRegressor(n_estimators=10, random_state=42)
    >>> et_reg.fit(X, y)
    >>> iforest = convert_extratrees_to_isolationforest(et_reg)
    
    For ONNX export (requires max_features=None):
    >>> et_onnx = ExtraTreesClassifier(n_estimators=10, max_features=None)
    >>> et_onnx.fit(X, y)
    >>> iforest_onnx = convert_extratrees_to_isolationforest(et_onnx)
    >>> # Now can export to ONNX
    >>> from skl2onnx import to_onnx
    >>> onnx_model = to_onnx(iforest_onnx, X[:1].astype(np.float32))
    
    Notes
    -----
    The converted IsolationForest inherits the tree structures from the ExtraTrees
    model, which were trained using supervised (classifier) or regression criteria.
    This differs from standard IsolationForest which builds trees using random splits
    on random features. The anomaly scores may therefore differ from a natively
    trained IsolationForest, but can be useful for repurposing existing models.
    
    See Also
    --------
    sklearn.ensemble.IsolationForest : The target model class
    sklearn.ensemble.ExtraTreesClassifier : Source classifier model
    sklearn.ensemble.ExtraTreesRegressor : Source regressor model
    """
    # Validate input type
    if not isinstance(extratrees_model, (ExtraTreesClassifier, ExtraTreesRegressor)):
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
    
    # Extract parameters from the ExtraTrees model
    n_estimators = len(extratrees_model.estimators_)
    max_samples = extratrees_model.max_samples
    max_features = extratrees_model.max_features
    bootstrap = extratrees_model.bootstrap
    n_jobs = extratrees_model.n_jobs
    random_state = extratrees_model.random_state
    verbose = extratrees_model.verbose
    
    # Create a new IsolationForest object with similar parameters
    iforest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples if max_samples is not None else 'auto',
        contamination=contamination,
        max_features=max_features if max_features is not None else 1.0,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=False
    )
    
    # Copy the tree estimators from ExtraTrees to IsolationForest
    # Deep copy to avoid modifying the original model
    iforest.estimators_ = [copy.deepcopy(tree) for tree in extratrees_model.estimators_]
    
    # Set feature information
    iforest.n_features_in_ = extratrees_model.n_features_in_
    if hasattr(extratrees_model, 'feature_names_in_'):
        iforest.feature_names_in_ = extratrees_model.feature_names_in_
    
    # Copy estimators_features_ - which features each estimator uses
    if hasattr(extratrees_model, 'estimators_features_'):
        iforest.estimators_features_ = extratrees_model.estimators_features_
    else:
        # If not available, assume all features are used by all estimators
        iforest.estimators_features_ = [
            np.arange(extratrees_model.n_features_in_) 
            for _ in range(len(iforest.estimators_))
        ]
    
    # Set _max_features (internal attribute required by IsolationForest)
    if max_features is None:
        iforest._max_features = extratrees_model.n_features_in_
    elif isinstance(max_features, str):
        if max_features == 'sqrt':
            iforest._max_features = int(np.sqrt(extratrees_model.n_features_in_))
        elif max_features == 'log2':
            iforest._max_features = int(np.log2(extratrees_model.n_features_in_))
        else:
            iforest._max_features = extratrees_model.n_features_in_
    elif isinstance(max_features, int):
        iforest._max_features = max_features
    elif isinstance(max_features, float):
        iforest._max_features = int(max_features * extratrees_model.n_features_in_)
    else:
        iforest._max_features = extratrees_model.n_features_in_
    
    # Calculate max_samples_ and _max_samples for score computation
    if max_samples is None:
        iforest.max_samples_ = 256
        iforest._max_samples = 256
    elif isinstance(max_samples, int):
        iforest.max_samples_ = max_samples
        iforest._max_samples = max_samples
    elif isinstance(max_samples, float):
        # Would need training data to compute, so we set a reasonable default
        iforest.max_samples_ = 256
        iforest._max_samples = 256
    else:
        iforest.max_samples_ = 256
        iforest._max_samples = 256
    
    # Pre-compute decision path lengths and average path lengths for each tree
    # This is required for the IsolationForest score_samples method
    _average_path_length_per_tree = []
    _decision_path_lengths = []
    
    for tree in iforest.estimators_:
        # Compute average path length for each node based on number of samples
        avg_path = _average_path_length(tree.tree_.n_node_samples)
        _average_path_length_per_tree.append(avg_path)
        
        # Compute depth of each node
        node_depths = tree.tree_.compute_node_depths()
        _decision_path_lengths.append(node_depths)
    
    iforest._average_path_length_per_tree = _average_path_length_per_tree
    iforest._decision_path_lengths = _decision_path_lengths
    
    # Set offset for decision function
    # The offset is used to center the decision function at 0 for the threshold
    if offset is not None:
        iforest.offset_ = offset
    else:
        # Default offset similar to IsolationForest with contamination='auto'
        iforest.offset_ = -0.5
    
    return iforest


def _average_path_length(n_samples_leaf):
    """
    Compute the average path length in an isolation tree.
    
    This is a helper function used internally by the conversion process.
    The average path length in a n_samples iTree is equal to the average
    path length of an unsuccessful BST search since the latter has the
    same structure as an isolation tree.
    
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimator.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
        The average path length for each node.
        
    Notes
    -----
    This function implements the formula from the original Isolation Forest paper:
    - c(n) = 2H(n-1) - 2(n-1)/n for n > 2
    - c(2) = 1
    - c(1) = 0
    where H(i) is the harmonic number, approximated as ln(i) + gamma
    """
    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)


# Public API exports
__all__ = ['convert_extratrees_to_isolationforest']

