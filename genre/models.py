"""The three classifiers being compared."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 1


def k_nearest_neighbours(n_neighbors: int = 15) -> Pipeline:
    """KNN behind a scaler.

    Distance-based, so an unscaled feature dominates purely by its units:
    duration in milliseconds runs to six figures while danceability sits in
    [0, 1], and without scaling the neighbourhood is decided by duration alone.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def decision_tree(max_depth: int = 9, min_samples_leaf: int = 60) -> DecisionTreeClassifier:
    """A single tree. Splits on thresholds, so scaling makes no difference."""
    return DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED
    )


def random_forest(
    n_estimators: int = 25, max_depth: int = 10, min_samples_leaf: int = 1
) -> RandomForestClassifier:
    """Many trees on bootstrapped samples and random feature subsets."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED,
        n_jobs=-1,
    )


BUILDERS = {
    "knn": k_nearest_neighbours,
    "decision tree": decision_tree,
    "random forest": random_forest,
}
