#!/usr/bin/env python
"""Evaluate MERGE on measured external materials without retraining."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from merge.data.external import build_external_targets, preprocess_external_data
from merge.data.metal_features import (
    ALL_COLS,
    aggregate_features_by_metal_class,
    apply_scaler,
    load_stats,
)
from merge.data.pseudobulk import load_pseudobulk
from merge.evaluation.bootstrap import bootstrap_group_confidence_intervals
from merge.evaluation.metrics import compute_all_metrics, compute_metrics_from_confusion
from merge.evaluation.topk_direction import compute_topk_direction_accuracy
from merge.inference import MaterialControlDataset, flatten_group_predictions, run_material_inference
from merge.models.full_model import build_model
from merge.utils.config import load_config
from merge.utils.gpu import get_device
from merge.utils.io import save_json
from merge.utils.logging import setup_logger
from merge.utils.seed import set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prepare_external_features(csv_path, material_specs, scaler_path):
    table = pd.read_csv(csv_path)
    feature_classes = [str(spec["feature_class"]) for spec in material_specs]
    if len(feature_classes) != len(set(feature_classes)):
        raise ValueError("Each external material must reference a unique feature_class")
    selected = table[table["metal_class"].astype(str).isin(feature_classes)].copy()
    missing = set(feature_classes) - set(selected["metal_class"].astype(str))
    if missing:
        raise ValueError(f"Descriptor table is missing external classes: {sorted(missing)}")
    aggregated = aggregate_features_by_metal_class(selected).set_index("metal_class")
    ordered = aggregated.loc[feature_classes].reset_index()
    ordered.insert(0, "material", [str(spec["name"]) for spec in material_specs])
    standardized = apply_scaler(ordered, load_stats(str(scaler_path)))
    for index, column in enumerate(ALL_COLS):
        ordered[f"{column}_standardized"] = standardized[:, index]
    return ordered, standardized


def _metric_subsets(flat, field):
    values = np.asarray(flat[field]).astype(str)
    return {
        value: compute_all_metrics(
            flat["y_true"][values == value],
            flat["y_pred"][values == value],
            flat["y_prob"][values == value],
        )
        for value in sorted(set(values))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-config", required=True, help="Material manifest YAML")
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs/model_config.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs/training_config.yaml"))
    parser.add_argument("--eval-config", default=str(PROJECT_ROOT / "configs/eval_config.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    eval_config = load_config(args.eval_config)
    external_config = load_config(args.external_config)
    material_specs = [dict(spec) for spec in external_config.materials]
    if not material_specs:
        raise ValueError("The external manifest contains no materials")
    run_name = str(args.run_name or training_config.run_name)
    seed = int(args.seed if args.seed is not None else training_config.seed)
    set_global_seed(seed)
    output_root = Path(str(data_config.output_root))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "external" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("external_evaluation", str(output_dir / "external_evaluation.log"))
    checkpoint = args.checkpoint or str(output_root / "checkpoints" / run_name / "best.pt")
    scaler_path = output_root / "preprocessing" / f"metal_feature_stats_{run_name}.json"
    feature_table, standardized = _prepare_external_features(
        str(external_config.metal_feature_csv), material_specs, scaler_path
    )
    feature_table.to_csv(output_dir / "external_features_aggregated_standardized.csv", index=False)

    gene_table = pd.read_csv(output_root / "preprocessing" / "gene_index.csv")
    training_genes = gene_table["gene_name"].astype(str).tolist()
    if len(training_genes) != len(set(training_genes)):
        raise ValueError("The frozen training gene index contains duplicates")
    processed, available, qc_summary = preprocess_external_data(
        material_specs, training_genes, data_config, logger
    )
    qc_summary.to_csv(output_dir / "external_data_qc_summary.csv", index=False)
    control = ad.read_h5ad(output_root / "processed" / f"{data_config.ctrl_class}_processed.h5ad")
    if control.var_names.astype(str).tolist() != training_genes:
        raise ValueError("Control gene order differs from the frozen training gene index")
    control_pb, control_meta = load_pseudobulk(
        str(output_root / "pseudobulk" / run_name / "ctrl_pseudobulk.npz")
    )
    cell_types = control_meta["cell_type"].astype(str).tolist()
    if sorted(control.obs["cell_type"].astype(str).unique()) != cell_types:
        raise ValueError("Control pseudo-bulk cell-type order is inconsistent")
    labels, responses, group_cells = build_external_targets(
        processed,
        available,
        control_pb,
        cell_types,
        threshold=float(training_config.labels.threshold),
        min_cells=int(training_config.labels.min_cells_per_group),
        ignore_index=int(training_config.labels.ignore_index),
    )
    external_ignore = int(training_config.labels.ignore_index)
    valid_groups = (labels != external_ignore).any(axis=2)
    inference_sizes = np.where(valid_groups, group_cells, 0)
    materials = [str(spec["name"]) for spec in material_specs]
    dataset = MaterialControlDataset(
        control, standardized, materials, cell_types, inference_sizes, seed
    )
    if len(dataset) == 0:
        raise ValueError("No external groups meet the minimum cell threshold")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(
        model_config,
        n_genes=len(training_genes),
        gene_names=training_genes,
        checkpoint_path=checkpoint,
    )
    device = get_device(True)
    model.to(device)
    inference = run_material_inference(
        model,
        loader,
        len(materials),
        len(cell_types),
        len(training_genes),
        device,
        labels=labels,
        logger=logger,
        ignore_index=int(training_config.labels.ignore_index),
    )
    flat = flatten_group_predictions(
        labels,
        inference["predictions"],
        inference["mean_probabilities"],
        materials,
        cell_types,
        ignore_index=int(training_config.labels.ignore_index),
    )
    metrics = compute_all_metrics(flat["y_true"], flat["y_pred"], flat["y_prob"])
    unchanged = compute_all_metrics(flat["y_true"], np.ones_like(flat["y_true"]))
    by_material = _metric_subsets(flat, "material")
    by_celltype = _metric_subsets(flat, "cell_type")
    topk = {}
    k_list = [int(k) for k in eval_config.topk.k_list]
    for material_index, material in enumerate(materials):
        topk[material] = compute_topk_direction_accuracy(
            responses[material_index],
            labels[material_index],
            inference["predictions"][material_index],
            k_list=k_list,
        )
    for k in k_list:
        values = [topk[material][k]["overall"] for material in materials]
        values = [value for value in values if value is not None]
        metrics[f"top{k}_dir_acc"] = float(np.mean(values)) if values else None
    intervals = bootstrap_group_confidence_intervals(
        flat["group_confusions"],
        n_bootstrap=int(eval_config.bootstrap.n_resamples),
        seed=seed,
    )
    singlecell = compute_metrics_from_confusion(inference["singlecell_confusion"])
    singlecell_by_material = {
        material: compute_metrics_from_confusion(inference["singlecell_confusion_by_material"][index])
        for index, material in enumerate(materials)
        if inference["singlecell_confusion_by_material"][index].sum() > 0
    }

    save_json(metrics, str(output_dir / "metrics_pseudobulk.json"))
    save_json(unchanged, str(output_dir / "metrics_unchanged_only_baseline.json"))
    save_json(by_material, str(output_dir / "metrics_by_material.json"))
    save_json(by_celltype, str(output_dir / "metrics_by_celltype.json"))
    save_json(singlecell, str(output_dir / "metrics_singlecell.json"))
    save_json(singlecell_by_material, str(output_dir / "metrics_singlecell_by_material.json"))
    save_json(topk, str(output_dir / "topk_direction_accuracy.json"))
    save_json(intervals, str(output_dir / "bootstrap_95ci.json"))
    save_json(
        {
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "scaler": str(scaler_path),
            "seed": seed,
            "retraining_or_fine_tuning": False,
            "materials": material_specs,
            "valid_groups": int(valid_groups.sum()),
            "label_threshold": float(training_config.labels.threshold),
            "min_cells_per_group": int(training_config.labels.min_cells_per_group),
        },
        str(output_dir / "evaluation_provenance.json"),
    )
    np.savez_compressed(
        output_dir / "external_ground_truth.npz",
        labels=labels,
        log2fc=responses,
        group_cell_counts=group_cells,
        materials=np.asarray(materials),
        cell_types=np.asarray(cell_types),
        genes=np.asarray(training_genes),
    )
    np.savez_compressed(
        output_dir / "external_pseudobulk_predictions.npz",
        labels=labels,
        predictions=inference["predictions"],
        probabilities=inference["mean_probabilities"],
        vote_counts=inference["vote_counts"],
        materials=np.asarray(materials),
        cell_types=np.asarray(cell_types),
        genes=np.asarray(training_genes),
    )
    rows = []
    for scope, values in [("external_pooled", metrics), *by_material.items()]:
        rows.append(
            {
                "scope": scope,
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                "balanced_accuracy": values["balanced_accuracy"],
                "mcc": values["mcc"],
                "n_labels": values["n_samples"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "external_core_metrics.csv", index=False)
    logger.info("External evaluation complete: macro_f1=%.6f, groups=%d", metrics["macro_f1"], valid_groups.sum())


if __name__ == "__main__":
    main()
