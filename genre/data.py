"""Loading and preparing the track dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "music_genre"
# Free-text identifiers. Keeping them would let a model memorise which artist
# makes which genre rather than learning anything about the audio.
IDENTIFIERS = ("artist_name", "track_name")
CATEGORICAL = ("key", "mode")


@dataclass(frozen=True)
class Dataset:
    features: pd.DataFrame
    labels: np.ndarray
    classes: list[str]

    def __len__(self) -> int:
        return len(self.features)


def load(path: Path | str, target: str = TARGET) -> Dataset:
    """Read the CSV, drop identifiers, encode categoricals, fill gaps.

    Missing numeric values are filled with the column median rather than dropped.
    Three columns have gaps covering roughly a tenth of the rows between them, and
    dropping those rows would discard far more signal than the gaps themselves
    represent. The median is used over the mean because these distributions are
    skewed, and a few very long tracks would drag the mean away from typical.
    """
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"{path} has no {target!r} column")

    frame = frame.drop(columns=[c for c in IDENTIFIERS if c in frame.columns])
    frame = frame.dropna(subset=[target])

    labels_text = frame[target].astype(str)
    classes = sorted(labels_text.unique())
    labels = labels_text.map({name: i for i, name in enumerate(classes)}).to_numpy()

    features = frame.drop(columns=[target])
    for column in CATEGORICAL:
        if column in features.columns:
            features[column] = features[column].astype("category").cat.codes

    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True))
    return Dataset(features=features, labels=labels, classes=classes)


def split(dataset: Dataset, test_size: float = 0.33, seed: int = 1):
    """Stratified split, so every genre keeps its share in both halves.

    Stratifying matters even on a balanced dataset: a random split of six classes
    still drifts by a percentage point or two, and that drift lands directly in
    the reported accuracy.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        dataset.features,
        dataset.labels,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.labels,
    )
