from __future__ import annotations     # just to avoid lazy strings when some object refers to itself while initialization

from sklearn.metrics import precision_recall_curve
import pandas as pd
from src.metrics import evaluate_predictions, save_metrics
from dataclasses import dataclass, field
from typing import Optional
import math
import random
import numpy as np


# internal tree node
@dataclass
class _IsolationNode:
    # leaf settings
    size: int=0            # number of samples

    # internal node settings
    feature: Optional[int] = None       # what feature is used to split the data
    threshold: Optional[float] = None     # what was the threshold for splitting
    left: Optional[_IsolationNode] = None
    right: Optional[_IsolationNode] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

@dataclass
class _IsolationTree:
    # required arguments
    max_depth: int
    rng: random.Random

    # internal state attributes, not required
    root: Optional[_IsolationNode] = field(init=False, default=None)
    n_features: Optional[int] = field(init=False, default=None)

    def fit(self, X: np.ndarray) -> _IsolationTree:
        self.n_features = X.shape[1]
        self.root = self._grow(X, depth=0)
        return self
    
    def _grow(self, X: np.ndarray, depth: int) -> _IsolationNode:
        n = len(X)           # this is not overall size but it shortens as we split more data
        if n <= 1 or depth >= self.max_depth:
            return _IsolationNode(size=n)
        
        feat = self.rng.randint(0, self.n_features-1)
        col = X[:, feat]
        lo, hi = col.min(), col.max()
        
        # if everything is the same, we can't split that
        if lo == hi:
            return _IsolationNode(size=n)

        threshold = self.rng.uniform(lo, hi)
        mask = col < threshold  # creates a list of True/False ([True, False, True...]) based on column values and threshold
        X_left, X_right = X[mask], X[~mask]

        node = _IsolationNode(feature=feat, threshold=threshold)
        node.left = self._grow(X_left, depth+1)
        node.right = self._grow(X_right, depth+1)
        return node

    # just traversing the tree and calculating the depth
    def path_length(self, x:np.ndarray) -> float:
        if self.root is None:
            raise ValueError("The tree must be fitted first before calculating its path")
        
        node = self.root
        depth = 0
        while not node.is_leaf:
            depth += 1
            if x[node.feature] < node.threshold:
                node = node.left 
            else:
                node = node.right
        
        # the leaf node that we stopped might have other data points (hit by max_depth) and would be unfair if we didn't count them. _c is an average path length of unsuccessful search in BST
        return depth + _c(node.size)

# The helper function to estimate average path
def _c(n: int) -> float:
    if n<= 1:
        return 0.0
    if n == 2:
        return 1.0
    
    # approximation of the (n-1)th harmonic number, using Euler-Mascheroni constant
    harmonic_number = np.log(n - 1) + 0.5772156649
    return 2.0 * harmonic_number - (2.0 * (n - 1) / n)

@dataclass
class IsolationForest:
    n_estimators: int = 100
    max_samples: int | str = "auto"
    contamination: float | str = 0.1
    max_features: float | int = 1.0
    random_state: int | None = None

    # internal state attributes needed for algorithm to work
    _trees: list = field(init=False, default_factory=list)
    _threshold: float = field(init=False, default=0.5)
    _psi: int = field(init=False, default=256)
    _n_features_to_use: int = field(init=False, default=0)

    def fit(self, X: np.ndarray) -> IsolationForest:
        X = np.asarray(X, dtype=float)   # converting the data to np array of floating-point numbers
        n, n_features = X.shape

        if self.max_samples == "auto":
            psi = min(256, n)
        else:
            psi = int(self.max_samples)
        self._psi = psi
        
        # it mirrors sklearn: 1/n_samples, floored at 0.001
        if self.contamination == "auto":
            contamination = min(0.1, max(0.001, 1.0 / n))
        else:
            contamination = float(self.contamination)

        # max_features: fraction of features to use per tree
        if isinstance(self.max_features, float):
            n_features_to_use = max(1, int(self.max_features * n_features))
        else:
            n_features_to_use = int(self.max_features)
        self._n_features_to_use = n_features_to_use

        max_depth = math.ceil(math.log2(psi)) if psi > 1 else 1

        rng = random.Random(self.random_state)
        self._trees = []

        for _ in range(self.n_estimators):
            idx = rng.sample(range(n), min(psi, n))
            X_sub = X[idx]

            # subsample features for this tree
            feat_idx = rng.sample(range(n_features), n_features_to_use)
            X_sub_feats = X_sub[:, feat_idx]

            tree = _IsolationTree(max_depth=max_depth, rng=rng)
            tree.fit(X_sub_feats)
            self._trees.append((tree, feat_idx))

        scores = self.score_samples(X)
        self._threshold = float(np.percentile(scores, 100 * (1 - contamination)))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mean_lengths = np.array([
            np.mean([tree.path_length(x[feat_idx]) for tree, feat_idx in self._trees])
            for x in X
        ])
        c_psi = _c(self._psi)
        if c_psi == 0:
            return np.ones(len(X)) * 0.5

        return 2.0 ** (-mean_lengths / c_psi)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._threshold - self.score_samples(X)

    def predict(self, X: np.ndarray) ->np.ndarray:
        # decision >= means that score >= threshold, so it should be anomalous
        return np.where(self.decision_function(X) >= 0, 1, -1)

def best_threshold_by_f1(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    return thresholds[np.argmax(f1_scores[:-1])]


def train_isolation_forest():
    X_train = pd.read_csv('data/processed/X_train_normal.csv').values
    X_val   = pd.read_csv('data/processed/X_val.csv').values
    X_test  = pd.read_csv('data/processed/X_test.csv').values
    y_val   = pd.read_csv('data/processed/y_val.csv').values.flatten()
    y_test  = pd.read_csv('data/processed/y_test.csv').values.flatten()

    clf = IsolationForest(
        n_estimators=100,
        max_samples=1024,
        contamination="auto",
        max_features=1.0,
        random_state=42
    )
    clf.fit(X_train)

    val_scores = clf.score_samples(X_val)
    threshold = best_threshold_by_f1(y_val, val_scores)

    test_scores = clf.score_samples(X_test)
    metrics = evaluate_predictions(y_test, test_scores, threshold)

    print(f"Isolation Forest: {metrics}")
    save_metrics(metrics, model_name="IsolationForest")