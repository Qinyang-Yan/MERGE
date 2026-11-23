#!/usr/bin/env python
"""Build pseudo-bulk expression targets without fitting model-input scalers."""

import argparse
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from merge.data.labels import assign_direction_labels, compute_label_distribution, compute_log2fc
from merge.data.pseudobulk import build_ctrl_pseudobulk, compute_pseudobulk, save_pseudobulk
from merge.utils.config import load_config
from merge.utils.io import ensure_dir, save_json
from merge.utils.logging import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs/training_config.yaml"))
    parser.add_argument("--overwrite", action="store_true", help="Replace this run's pseudo-bulk directory")
    args = parser.parse_args()
    data_config = load_config(args.data_config)
    training_config = load_config(args.training_config)
    output_root = Path(str(data_config.output_root))
    run_name = str(training_config.run_name)
    logger = setup_logger("pseudobulk", str(output_root / "logs" / f"{run_name}_pseudobulk.log"))
    processed_dir = output_root / "processed"
    pseudobulk_dir = output_root / "pseudobulk" / run_name
    if pseudobulk_dir.exists() and any(pseudobulk_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Pseudo-bulk directory exists: {pseudobulk_dir}; use --overwrite to rebuild it"
            )
        shutil.rmtree(pseudobulk_dir)
    ensure_dir(str(pseudobulk_dir))
    preprocessing_dir = ensure_dir(str(output_root / "preprocessing"))
    control = ad.read_h5ad(processed_dir / f"{data_config.ctrl_class}_processed.h5ad")
    control_pb, cell_types = build_ctrl_pseudobulk(control)
    control_meta = pd.DataFrame(
        {
            "cell_type": cell_types,
            "n_cells": [
                int((control.obs["cell_type"].astype(str) == cell_type).sum())
                for cell_type in cell_types
            ],
        }
    )
    save_pseudobulk(control_pb, control_meta, str(pseudobulk_dir / "ctrl_pseudobulk.npz"))
    pd.Series(cell_types, name="cell_type").to_json(
        preprocessing_dir / "celltype_mapping.json", force_ascii=False, indent=2
    )
    labels_by_sample = {}
    min_cells = int(training_config.labels.min_cells_per_group)
    threshold = float(training_config.labels.threshold)
    ignore_index = int(training_config.labels.ignore_index)
    for metal in data_config.metal_classes:
        adata = ad.read_h5ad(processed_dir / f"{metal}_processed.h5ad")
        matrix, metadata = compute_pseudobulk(
            adata,
            group_keys=["cell_type", "sample_id"],
            min_cells_per_group=min_cells,
        )
        if metadata.empty:
            logger.warning("%s has no groups with at least %d cells", metal, min_cells)
            continue
        metadata["sample_id"] = metadata["sample_id"].astype(str)
        metadata["metal_class"] = str(metal)
        feature_map = adata.obs[["sample_id", "metal_feature_id"]].copy()
        feature_map["sample_id"] = feature_map["sample_id"].astype(str)
        feature_map["metal_feature_id"] = feature_map["metal_feature_id"].astype(str)
        feature_map = feature_map.drop_duplicates()
        if feature_map["sample_id"].duplicated().any():
            raise ValueError(f"{metal} contains samples mapped to multiple descriptor IDs")
        metadata = metadata.merge(feature_map, on="sample_id", how="left", validate="many_to_one")
        for sample_id in metadata["sample_id"].unique():
            sample_meta = metadata[metadata["sample_id"] == sample_id]
            sample_matrix = np.full_like(control_pb, np.nan, dtype=np.float32)
            valid_groups = np.zeros(len(cell_types), dtype=bool)
            for row_index, row in sample_meta.iterrows():
                cell_type = str(row["cell_type"])
                if cell_type not in cell_types:
                    raise KeyError(f"Cell type {cell_type!r} is absent from the control reference")
                cell_type_index = cell_types.index(cell_type)
                sample_matrix[cell_type_index] = matrix[row_index]
                valid_groups[cell_type_index] = True
            filled = np.where(np.isnan(sample_matrix), control_pb, sample_matrix)
            response = compute_log2fc(filled, control_pb)
            labels = assign_direction_labels(response, threshold=threshold)
            labels[~valid_groups] = ignore_index
            if not valid_groups.any():
                continue
            save_pseudobulk(
                filled,
                pd.DataFrame(
                    {"cell_type": cell_types, "sample_id": sample_id, "valid": valid_groups}
                ),
                str(pseudobulk_dir / f"pert_pseudobulk_{sample_id}.npz"),
            )
            np.savez_compressed(
                pseudobulk_dir / f"labels_{sample_id}.npz",
                labels=labels,
                log2fc=response,
                celltypes=np.asarray(cell_types),
            )
            labels_by_sample[str(sample_id)] = labels
    if not labels_by_sample:
        raise ValueError("No valid pseudo-bulk labels were constructed")
    save_json(
        compute_label_distribution(labels_by_sample),
        str(preprocessing_dir / f"label_distribution_{run_name}.json"),
    )
    logger.info("Constructed pseudo-bulk targets for %d biological samples", len(labels_by_sample))


if __name__ == "__main__":
    main()
