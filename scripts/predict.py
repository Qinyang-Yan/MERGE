#!/usr/bin/env python
"""Predict gene-response directions for new materials without exposed-cell expression."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from merge.data.metal_features import ALL_COLS, aggregate_features_by_metal_class, apply_scaler, load_stats
from merge.data.pseudobulk import load_pseudobulk
from merge.inference import MaterialControlDataset, run_material_inference
from merge.models.full_model import build_model
from merge.utils.config import load_config
from merge.utils.gpu import get_device
from merge.utils.io import save_json
from merge.utils.logging import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metal-features", required=True, help="CSV of new-material replicate descriptors")
    parser.add_argument("--n-control-cells-per-type", required=True, type=int)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs/model_config.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs/training_config.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.n_control_cells_per_type < 1 or args.batch_size < 1:
        raise ValueError("Control-cell count and batch size must be positive")
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    run_name = str(args.run_name or training_config.run_name)
    seed = int(args.seed if args.seed is not None else training_config.seed)
    output_root = Path(str(data_config.output_root))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "predictions" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("predict", str(output_dir / "prediction.log"))
    checkpoint = args.checkpoint or str(output_root / "checkpoints" / run_name / "best.pt")
    scaler_path = output_root / "preprocessing" / f"metal_feature_stats_{run_name}.json"

    descriptors = aggregate_features_by_metal_class(pd.read_csv(args.metal_features))
    materials = descriptors["metal_class"].astype(str).tolist()
    standardized = apply_scaler(descriptors, load_stats(str(scaler_path)))
    for index, column in enumerate(ALL_COLS):
        descriptors[f"{column}_standardized"] = standardized[:, index]
    descriptors.to_csv(output_dir / "standardized_material_features.csv", index=False)
    training_genes = pd.read_csv(output_root / "preprocessing" / "gene_index.csv")["gene_name"].astype(str).tolist()
    control = ad.read_h5ad(output_root / "processed" / f"{data_config.ctrl_class}_processed.h5ad")
    if control.var_names.astype(str).tolist() != training_genes:
        raise ValueError("Control gene order differs from the frozen training gene index")
    _, control_meta = load_pseudobulk(str(output_root / "pseudobulk" / run_name / "ctrl_pseudobulk.npz"))
    cell_types = control_meta["cell_type"].astype(str).tolist()
    group_sizes = np.full(
        (len(materials), len(cell_types)), args.n_control_cells_per_type, dtype=np.int64
    )
    dataset = MaterialControlDataset(
        control, standardized, materials, cell_types, group_sizes, seed
    )
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
        logger=logger,
    )
    np.savez_compressed(
        output_dir / "merge_predictions.npz",
        predictions=inference["predictions"],
        probabilities=inference["mean_probabilities"],
        vote_counts=inference["vote_counts"],
        materials=np.asarray(materials),
        cell_types=np.asarray(cell_types),
        genes=np.asarray(training_genes),
        direction_classes=np.asarray(["Down", "Unchanged", "Up"]),
    )
    save_json(
        {
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "scaler": str(scaler_path),
            "seed": seed,
            "n_control_cells_per_material_cell_type": args.n_control_cells_per_type,
            "aggregation": "majority vote and mean probability across sampled control cells",
            "materials": materials,
            "cell_types": cell_types,
            "n_genes": len(training_genes),
        },
        str(output_dir / "prediction_provenance.json"),
    )
    logger.info("Saved predictions for %d materials and %d cell types", len(materials), len(cell_types))


if __name__ == "__main__":
    main()
