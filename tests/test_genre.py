"""Tests for loading, splitting, models and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genre.data import IDENTIFIERS, TARGET, load, split
from genre.evaluate import confusion, overfitting_gap, score
from genre.models import decision_tree, k_nearest_neighbours, random_forest


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = 120
    frame = pd.DataFrame(
        {
            "artist_name": ["someone"] * rows,
            "track_name": ["a song"] * rows,
            "popularity": rng.uniform(0, 100, rows),
            "danceability": rng.uniform(0, 1, rows),
            "duration_ms": rng.uniform(1e5, 3e5, rows),
            "key": rng.choice(list("ABCDEFG"), rows),
            "mode": rng.choice(["Major", "Minor"], rows),
            TARGET: ["Rock", "Jazz", "Rap"] * (rows // 3),
        }
    )
    # Gaps in two numeric columns, as the real dataset has.
    frame.loc[:9, "duration_ms"] = np.nan
    frame.loc[10:14, "popularity"] = np.nan
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)
    return path


def test_identifier_columns_are_dropped(csv):
    dataset = load(csv)
    assert not set(IDENTIFIERS) & set(dataset.features.columns)


def test_classes_are_sorted_and_labels_are_indices(csv):
    dataset = load(csv)
    assert dataset.classes == ["Jazz", "Rap", "Rock"]
    assert set(np.unique(dataset.labels)) == {0, 1, 2}


def test_every_feature_ends_up_numeric(csv):
    features = load(csv).features
    assert all(np.issubdtype(dtype, np.number) for dtype in features.dtypes)


def test_missing_values_are_filled_not_dropped(csv):
    dataset = load(csv)
    assert len(dataset) == 120
    assert not dataset.features.isna().any().any()


def test_filling_uses_the_median(csv):
    """The median, not the mean: these distributions are skewed."""
    dataset = load(csv)
    raw = pd.read_csv(csv)["duration_ms"]
    assert dataset.features["duration_ms"].iloc[0] == pytest.approx(raw.median())


def test_a_missing_target_column_is_an_error(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load(path)


def test_the_split_preserves_class_proportions(csv):
    dataset = load(csv)
    x_train, x_test, y_train, y_test = split(dataset, test_size=0.25, seed=0)
    assert len(x_train) + len(x_test) == len(dataset)
    for label in np.unique(dataset.labels):
        overall = (dataset.labels == label).mean()
        assert abs((y_train == label).mean() - overall) < 0.1
        assert abs((y_test == label).mean() - overall) < 0.1


def test_knn_scales_its_features_first():
    """Distance-based, so an unscaled column would dominate by its units alone."""
    assert "scale" in dict(k_nearest_neighbours().named_steps)


@pytest.mark.parametrize(
    "builder", [k_nearest_neighbours, decision_tree, random_forest],
    ids=["knn", "tree", "forest"],
)
def test_every_model_fits_and_predicts_one_label_per_row(builder, csv):
    dataset = load(csv)
    x_train, x_test, y_train, y_test = split(dataset, test_size=0.25, seed=0)
    model = builder()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    assert predicted.shape == y_test.shape
    assert set(np.unique(predicted)) <= set(np.unique(dataset.labels))


def test_perfect_predictions_score_one():
    labels = np.array([0, 1, 2, 1])
    result = score(labels, labels)
    assert (result.accuracy, result.precision, result.recall, result.f1) == (1.0,) * 4


def test_scores_fall_when_predictions_are_wrong():
    actual = np.array([0, 1, 2])
    assert score(actual, np.array([0, 1, 1])).accuracy == pytest.approx(2 / 3)


def test_confusion_matrix_is_square_and_totals_the_samples():
    actual = np.array([0, 1, 2, 2])
    matrix = confusion(actual, np.array([0, 1, 2, 1]))
    assert matrix.shape == (3, 3)
    assert matrix.sum() == len(actual)


def test_overfitting_gap_is_the_difference_in_accuracy():
    train = score(np.array([0, 1]), np.array([0, 1]))
    test = score(np.array([0, 1]), np.array([0, 0]))
    assert overfitting_gap(train, test) == pytest.approx(0.5)
