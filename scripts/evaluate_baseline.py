#!/usr/bin/env python
"""Evaluate the training-majority reference on the identical held-out samples."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from merge.data.splitting import load_sample_split
from merge.evaluation.metrics import compute_all_metrics
from merge.models.baselines import MajorityClassPredictor
from merge.utils.config import load_config
from merge.utils.io import save_json
from merge.utils.logging import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs/training_config.yaml"))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    data_config = load_config(args.data_config)
    training_config = load_config(args.training_config)
    run_name = str(args.run_name or training_config.run_name)
    output_root = Path(str(data_config.output_root))
    logger = setup_logger("majority_baseline", str(output_root / "logs" / f"{run_name}_majority.log"))
    split = load_sample_split(str(output_root / "reports" / f"{run_name}_split.json"))
    labels = {}
    for path in sorted((output_root / "pseudobulk" / run_name).glob("labels_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            labels[path.stem.removeprefix("labels_")] = data["labels"]
    training_labels = [labels[sample_id] for sample_id in split["train_sample_ids"] if sample_id in labels]
    if not training_labels:
        raise ValueError("No training labels available for the majority baseline")
    predictor = MajorityClassPredictor().fit(np.stack(training_labels))
    true_parts, pred_parts = [], []
    for sample_id in split["test_sample_ids"]:
        if sample_id not in labels:
            logger.warning("No labels for held-out sample %s", sample_id)
            continue
        true = labels[sample_id]
        pred = predictor.predict(true.shape[0])
        valid = true != int(training_config.labels.ignore_index)
        true_parts.append(true[valid])
        pred_parts.append(pred[valid])
    if not true_parts:
        raise ValueError("No test labels available for the majority baseline")
    metrics = compute_all_metrics(np.concatenate(true_parts), np.concatenate(pred_parts))
    output_dir = output_root / "baselines" / run_name / "majority"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, str(output_dir / "metrics.json"))
    pd.DataFrame([{"run": run_name, "baseline": "training_majority", **metrics}]).to_csv(
        output_dir / "metrics.csv", index=False
    )
    logger.info("Majority baseline complete: macro_f1=%.6f", metrics["macro_f1"])


if __name__ == "__main__":
    main()
