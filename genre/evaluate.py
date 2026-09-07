"""Scoring a fitted classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def format(self) -> str:
        return (
            f"accuracy {self.accuracy:.3f}  precision {self.precision:.3f}  "
            f"recall {self.recall:.3f}  f1 {self.f1:.3f}"
        )


def score(actual: np.ndarray, predicted: np.ndarray) -> Scores:
    """Weighted averages across the six genres."""
    return Scores(
        accuracy=float(accuracy_score(actual, predicted)),
        precision=float(precision_score(actual, predicted, average="weighted", zero_division=0)),
        recall=float(recall_score(actual, predicted, average="weighted", zero_division=0)),
        f1=float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    )


def confusion(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return confusion_matrix(actual, predicted)


def overfitting_gap(train: Scores, test: Scores) -> float:
    """How much better the model does on data it has already seen."""
    return train.accuracy - test.accuracy
