"""Command line entry point: ``python -m genre.cli``"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone

from genre.data import load, split
from genre.evaluate import confusion, overfitting_gap, score
from genre.figures import confusion_heatmap, feature_importance, sweep
from genre.models import BUILDERS, decision_tree, k_nearest_neighbours, random_forest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.csv"
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


def _sweep(builder, values, x_train, y_train, x_test, y_test):
    train_scores, test_scores = [], []
    for value in values:
        model = clone(builder(value))
        model.fit(x_train, y_train)
        train_scores.append(float((model.predict(x_train) == y_train).mean()))
        test_scores.append(float((model.predict(x_test) == y_test).mean()))
    return {"values": list(values), "train": train_scores, "test": test_scores}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--skip-sweeps", action="store_true")
    args = parser.parse_args(argv)

    dataset = load(args.dataset)
    x_train, x_test, y_train, y_test = split(dataset)
    print(f"{len(dataset):,} tracks, {len(dataset.classes)} genres, "
          f"{x_train.shape[1]} features, {len(x_train):,} train / {len(x_test):,} test\n")

    summary = {}
    for name, builder in BUILDERS.items():
        model = builder()
        model.fit(x_train, y_train)
        train = score(y_train, model.predict(x_train))
        test = score(y_test, model.predict(x_test))
        summary[name] = {
            "train": vars(train), "test": vars(test),
            "overfitting_gap": round(overfitting_gap(train, test), 4),
        }
        print(f"{name:<15} train {train.accuracy:.3f}   test  {test.format()}")

    forest = random_forest()
    forest.fit(x_train, y_train)
    confusion_heatmap(
        confusion(y_test, forest.predict(x_test)), dataset.classes,
        args.docs / "confusion.png"
    )
    feature_importance(
        list(x_train.columns), forest.feature_importances_, args.docs / "importance.png"
    )

    if not args.skip_sweeps:
        curves = {
            "k neighbours": _sweep(lambda v: k_nearest_neighbours(v), range(1, 60, 4),
                                   x_train, y_train, x_test, y_test),
            "tree max depth": _sweep(lambda v: decision_tree(max_depth=v, min_samples_leaf=1),
                                     range(1, 26), x_train, y_train, x_test, y_test),
            "forest max depth": _sweep(lambda v: random_forest(max_depth=v),
                                       range(1, 26), x_train, y_train, x_test, y_test),
        }
        sweep(curves, args.docs / "sweeps.png")
        summary["sweeps"] = {k: {"best": v["values"][int(np.argmax(v["test"]))]}
                             for k, v in curves.items()}

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.results / 'metrics.json'} and figures to {args.docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
