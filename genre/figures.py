"""Figures: hyperparameter sweeps, the confusion matrix, feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN_COLOUR = "#cf222e"
TEST_COLOUR = "#1a7f37"


def sweep(curves: dict[str, dict], output_path: Path) -> Path:
    """One panel per swept hyperparameter, train against test accuracy."""
    figure, axes = plt.subplots(1, len(curves), figsize=(5.0 * len(curves), 3.9))
    axes = [axes] if len(curves) == 1 else list(axes)

    for axis, (name, curve) in zip(axes, curves.items()):
        axis.plot(curve["values"], curve["train"], label="train", color=TRAIN_COLOUR)
        axis.plot(curve["values"], curve["test"], label="test", color=TEST_COLOUR)
        best = curve["values"][int(np.argmax(curve["test"]))]
        axis.axvline(best, color="#57606a", linestyle="--", linewidth=1)
        axis.set_title(f"{name}\nbest test accuracy at {best}", fontsize=11)
        axis.set_xlabel(name)
        axis.set_ylabel("accuracy")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle("The gap between the two lines is memorisation", fontsize=12, y=1.04)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def confusion_heatmap(matrix: np.ndarray, classes: list[str], output_path: Path) -> Path:
    """Row-normalised confusion, so each row reads as a share of that genre."""
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(normalised, cmap="Greens", vmin=0, vmax=1)

    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("predicted"), axis.set_ylabel("actual")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, f"{normalised[i, j]:.2f}", ha="center", va="center",
                      fontsize=9, color="white" if normalised[i, j] > 0.5 else "#24292f")
    figure.colorbar(image, ax=axis, shrink=0.8)
    axis.set_title("Where the random forest confuses genres", fontsize=12)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def feature_importance(names: list[str], importances: np.ndarray, output_path: Path) -> Path:
    order = np.argsort(importances)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.barh([names[i] for i in order], importances[order], color="#0969da")
    axis.set_xlabel("importance")
    axis.set_title("What the random forest splits on", fontsize=12)
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path
