#!/usr/bin/env python
"""Evaluate a frozen MERGE checkpoint on the held-out biological samples."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from merge.data.dataset import SingleCellPerturbDataset
from merge.data.pairing import CellTypeStratifiedPairer
from merge.data.splitting import load_sample_split
from merge.evaluation.metrics import (
    compute_all_metrics,
    compute_metrics_from_confusion,
    compute_per_celltype_metrics,
    compute_per_metal_metrics,
)
from merge.evaluation.topk_direction import compute_topk_direction_accuracy
from merge.models.full_model import build_model
from merge.utils.config import load_config
from merge.utils.gpu import get_device
from merge.utils.io import save_json
from merge.utils.logging import setup_logger
from merge.utils.seed import seed_worker, set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_labels(directory: Path):
    labels, responses = {}, {}
    cell_types = None
    for path in sorted(directory.glob("labels_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            sample_id = path.stem.removeprefix("labels_")
            labels[sample_id] = data["labels"]
            responses[sample_id] = data["log2fc"]
            current = [str(value) for value in data["celltypes"]]
            if cell_types is None:
                cell_types = current
            elif current != cell_types:
                raise ValueError("Cell-type order differs across label artifacts")
    if not labels or cell_types is None:
        raise FileNotFoundError(f"No label artifacts found in {directory}")
    return labels, responses, cell_types


def _load_processed(data_config):
    root = Path(str(data_config.output_root)) / "processed"
    conditions = [str(data_config.ctrl_class), *map(str, data_config.metal_classes)]
    return {condition: ad.read_h5ad(root / f"{condition}_processed.h5ad") for condition in conditions}


def _make_test_dataset(adatas, metadata, split, features, feature_ids, labels, celltype_to_idx, ctrl_class):
    datasets = []
    test_ids = set(split["test_sample_ids"])
    test_metadata = metadata[metadata["sample_id"].isin(test_ids)]
    for metal, group in test_metadata.groupby("metal_class"):
        adata = adatas[str(metal)]
        sample_ids = set(group["sample_id"].astype(str))
        subset = adata[adata.obs["sample_id"].astype(str).isin(sample_ids)].copy()
        if subset.n_obs == 0:
            continue
        subset.uns["sample_id_order"] = list(feature_ids)
        datasets.append(
            SingleCellPerturbDataset(
                subset,
                adatas[ctrl_class],
                features,
                labels,
                celltype_to_idx,
                CellTypeStratifiedPairer(adatas[ctrl_class]),
            )
        )
    if not datasets:
        raise ValueError("No held-out cells match the frozen test partition")
    return ConcatDataset(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs/model_config.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs/training_config.yaml"))
    parser.add_argument("--eval-config", default=str(PROJECT_ROOT / "configs/eval_config.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    eval_config = load_config(args.eval_config)
    run_name = str(args.run_name or training_config.run_name)
    seed = int(training_config.seed)
    set_global_seed(seed)
    output_root = Path(str(data_config.output_root))
    logger = setup_logger(f"evaluate_{run_name}", str(output_root / "logs" / f"evaluate_{run_name}.log"))
    split = load_sample_split(str(output_root / "reports" / f"{run_name}_split.json"))
    checkpoint = args.checkpoint or str(output_root / "checkpoints" / run_name / "best.pt")
    adatas = _load_processed(data_config)
    labels, responses, cell_types = _load_labels(output_root / "pseudobulk" / run_name)
    metadata = pd.read_csv(output_root / "preprocessing" / "sample_metadata.csv", dtype=str)
    metadata = metadata[metadata["sample_id"].isin(labels)].copy()
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Each biological sample must have exactly one metadata row")
    sample_to_metal = metadata.set_index("sample_id")["metal_class"].to_dict()

    feature_path = output_root / "pseudobulk" / run_name / "metal_features_standardized.npz"
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Run-specific standardized descriptors not found: {feature_path}. "
            "MERGE never falls back to descriptors standardized by another run."
        )
    with np.load(feature_path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        feature_ids = [str(value) for value in data["sample_ids"]]
    ctrl_class = str(data_config.ctrl_class)
    control_cell_types = sorted(adatas[ctrl_class].obs["cell_type"].astype(str).unique())
    if control_cell_types != cell_types:
        raise ValueError("Control cell-type order does not match the label artifacts")
    celltype_to_idx = {cell_type: index for index, cell_type in enumerate(cell_types)}
    dataset = _make_test_dataset(
        adatas, metadata, split, features, feature_ids, labels, celltype_to_idx, ctrl_class
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.training.batch_size),
        shuffle=False,
        num_workers=int(training_config.training.num_workers),
        pin_memory=bool(training_config.training.pin_memory),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    gene_names = adatas[ctrl_class].var_names.astype(str).tolist()
    model = build_model(
        model_config, n_genes=len(gene_names), gene_names=gene_names, checkpoint_path=checkpoint
    )
    device = get_device(True)
    model.to(device).eval()

    test_ids = [sample_id for sample_id in split["test_sample_ids"] if sample_id in labels]
    vote_counts = {
        sample_id: np.zeros((*labels[sample_id].shape, 3), dtype=np.int32) for sample_id in test_ids
    }
    probability_sums = {
        sample_id: np.zeros((*labels[sample_id].shape, 3), dtype=np.float64) for sample_id in test_ids
    }
    group_counts = {
        sample_id: np.zeros(labels[sample_id].shape[0], dtype=np.int64) for sample_id in test_ids
    }
    singlecell_confusion = np.zeros((3, 3), dtype=np.int64)
    ignore_index = int(training_config.labels.ignore_index)
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["x_ctrl"].to(device), batch["metal_feat"].to(device))
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            predictions = logits.argmax(dim=-1).cpu().numpy()
            true = batch["labels"].numpy()
            cell_type_indices = batch["cell_type_idx"].numpy()
            sample_ids = batch["sample_id"]
            for row, sample_id in enumerate(sample_ids):
                sample_id = str(sample_id)
                cell_type_index = int(cell_type_indices[row])
                valid = true[row] != ignore_index
                if valid.any():
                    encoded = true[row, valid].astype(np.int64) * 3 + predictions[row, valid]
                    singlecell_confusion += np.bincount(encoded, minlength=9).reshape(3, 3)
                for class_index in range(3):
                    vote_counts[sample_id][cell_type_index, :, class_index] += (
                        predictions[row] == class_index
                    )
                probability_sums[sample_id][cell_type_index] += probabilities[row]
                group_counts[sample_id][cell_type_index] += 1

    pb_true, pb_pred, pb_prob = [], [], []
    pb_cell_types, pb_samples, pb_gene_indices = [], [], []
    topk_by_sample = {}
    for sample_id in test_ids:
        predictions = vote_counts[sample_id].argmax(axis=-1)
        probabilities = np.zeros_like(probability_sums[sample_id], dtype=np.float32)
        for cell_type_index in range(len(cell_types)):
            count = int(group_counts[sample_id][cell_type_index])
            valid = labels[sample_id][cell_type_index] != ignore_index
            if valid.any() and count == 0:
                raise ValueError(f"No predictions generated for {sample_id}/{cell_types[cell_type_index]}")
            if count:
                probabilities[cell_type_index] = (
                    probability_sums[sample_id][cell_type_index] / count
                ).astype(np.float32)
            if not valid.any():
                continue
            pb_true.append(labels[sample_id][cell_type_index, valid])
            pb_pred.append(predictions[cell_type_index, valid])
            pb_prob.append(probabilities[cell_type_index, valid])
            pb_cell_types.append(np.repeat(cell_types[cell_type_index], int(valid.sum())))
            pb_samples.append(np.repeat(sample_id, int(valid.sum())))
            pb_gene_indices.append(np.flatnonzero(valid))
        topk_by_sample[sample_id] = compute_topk_direction_accuracy(
            responses[sample_id],
            labels[sample_id],
            predictions,
            k_list=list(eval_config.topk.k_list),
        )
    if not pb_true:
        raise ValueError("No valid pseudo-bulk predictions were generated")
    true_flat = np.concatenate(pb_true)
    pred_flat = np.concatenate(pb_pred)
    prob_flat = np.concatenate(pb_prob)
    cell_type_flat = np.concatenate(pb_cell_types)
    sample_flat = np.concatenate(pb_samples)
    metal_flat = np.asarray([sample_to_metal[sample_id] for sample_id in sample_flat])
    metrics = compute_all_metrics(true_flat, pred_flat, prob_flat)
    singlecell_metrics = compute_metrics_from_confusion(singlecell_confusion)
    per_celltype = compute_per_celltype_metrics(true_flat, pred_flat, cell_type_flat)
    per_metal = compute_per_metal_metrics(true_flat, pred_flat, metal_flat)
    for k in eval_config.topk.k_list:
        per_material_values = {}
        for sample_id, result in topk_by_sample.items():
            value = result[int(k)]["overall"]
            if value is not None:
                per_material_values.setdefault(sample_to_metal[sample_id], []).append(value)
        material_means = [np.mean(values) for values in per_material_values.values()]
        metrics[f"top{int(k)}_dir_acc"] = float(np.mean(material_means)) if material_means else None

    evaluation_dir = output_root / "evaluation" / run_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, str(evaluation_dir / "metrics_pseudobulk.json"))
    save_json(singlecell_metrics, str(evaluation_dir / "metrics_singlecell.json"))
    save_json(per_celltype, str(evaluation_dir / "per_celltype_metrics.json"))
    save_json(per_metal, str(evaluation_dir / "per_metal_metrics.json"))
    save_json(topk_by_sample, str(evaluation_dir / "topk_direction_accuracy.json"))
    np.savez_compressed(
        evaluation_dir / "confusion_matrix.npz",
        pseudobulk=np.asarray(metrics["confusion_matrix"]),
        singlecell=singlecell_confusion,
    )
    if not args.no_save_predictions:
        np.savez_compressed(
            evaluation_dir / "per_gene_predictions.npz",
            y_true=true_flat,
            y_pred=pred_flat,
            proba=prob_flat.astype(np.float32),
            cell_type=cell_type_flat,
            sample_id=sample_flat,
            gene_idx=np.concatenate(pb_gene_indices),
            gene_names=np.asarray(gene_names),
        )
    pd.DataFrame([{"run": run_name, **metrics}]).to_csv(
        output_root / "reports" / f"{run_name}_metrics.csv", index=False
    )
    logger.info("Evaluation complete: macro_f1=%.6f, n=%d", metrics["macro_f1"], len(true_flat))


if __name__ == "__main__":
    main()
