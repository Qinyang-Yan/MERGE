#!/usr/bin/env python
"""Freeze sample partitions and fit the MERGE gene space on training data only."""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from merge.data.loader import load_all_data
from merge.data.preprocessing import preprocess_with_frozen_split
from merge.data.splitting import load_sample_split, make_sample_split, save_sample_split
from merge.utils.config import load_config, save_config
from merge.utils.io import ensure_dir, save_json, write_text
from merge.utils.logging import setup_logger
from merge.utils.seed import set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_metadata(adata_dict, metal_classes):
    rows = []
    for metal in metal_classes:
        adata = adata_dict[str(metal)]
        for sample_id, group in adata.obs.groupby("sample_id", observed=True, sort=True):
            if str(sample_id) == "unknown":
                raise ValueError("Unknown sample IDs must be rejected during loading")
            feature_ids = group["metal_feature_id"].astype(str).unique().tolist()
            if len(feature_ids) != 1:
                raise ValueError(f"{metal}/{sample_id} maps to multiple descriptors: {feature_ids}")
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "metal_class": str(metal),
                    "metal_feature_id": feature_ids[0],
                    "n_cells_before_qc": int(len(group)),
                }
            )
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError("No perturbation samples were discovered")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs/training_config.yaml"))
    parser.add_argument("--override", nargs="*", default=None, help="Data-config dot-list overrides")
    parser.add_argument("--overwrite", action="store_true", help="Replace processed files for this run")
    args = parser.parse_args()
    data_config = load_config(args.data_config, args.override)
    training_config = load_config(args.training_config)
    set_global_seed(int(training_config.seed))
    output_root = Path(str(data_config.output_root))
    run_name = str(training_config.run_name)
    logger = setup_logger("preprocess", str(output_root / "logs" / f"{run_name}_preprocess.log"))
    processed_dir = output_root / "processed"
    if processed_dir.exists() and any(processed_dir.glob("*_processed.h5ad")) and not args.overwrite:
        raise FileExistsError(
            f"Processed files already exist in {processed_dir}; use --overwrite to replace them"
        )
    if args.overwrite and processed_dir.exists():
        shutil.rmtree(processed_dir)
    ensure_dir(str(processed_dir))
    pre_dir = ensure_dir(str(output_root / "preprocessing"))
    report_dir = ensure_dir(str(output_root / "reports"))

    adata_dict, feature_table = load_all_data(data_config, logger)
    raw_metadata = _sample_metadata(adata_dict, data_config.metal_classes)
    split_path = report_dir / f"{run_name}_split.json"
    if split_path.exists() and not args.overwrite:
        split = load_sample_split(str(split_path))
        logger.info("Reusing the existing frozen split at %s", split_path)
    else:
        split_config = training_config.get("split", {})
        split = make_sample_split(
            raw_metadata,
            val_samples_per_metal=int(split_config.get("val_samples_per_metal", 1)),
            test_samples_per_metal=int(split_config.get("test_samples_per_metal", 1)),
        )
        save_sample_split(split, str(split_path))
        logger.info("Saved the frozen sample split to %s", split_path)

    available_samples = set(raw_metadata["sample_id"].astype(str))
    for partition, sample_ids in split.items():
        missing = set(map(str, sample_ids)) - available_samples
        if missing:
            raise ValueError(f"Frozen split partition {partition} references missing samples: {sorted(missing)}")

    processed = preprocess_with_frozen_split(adata_dict, data_config, split, logger)
    post_qc_rows = []
    summary_rows = []
    for metal, adata in processed.items():
        path = processed_dir / f"{metal}_processed.h5ad"
        adata.write_h5ad(path)
        summary_rows.append({"metal_class": metal, "n_cells": adata.n_obs, "n_genes": adata.n_vars})
        if str(metal) != str(data_config.ctrl_class):
            for sample_id, group in adata.obs.groupby("sample_id", observed=True, sort=True):
                feature_ids = group["metal_feature_id"].astype(str).unique().tolist()
                if len(feature_ids) != 1:
                    raise ValueError(f"{metal}/{sample_id} maps to multiple descriptors after QC")
                post_qc_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "metal_class": str(metal),
                        "metal_feature_id": feature_ids[0],
                        "n_cells": int(len(group)),
                    }
                )
        logger.info("Saved %s", path)
    first = next(iter(processed.values()))
    pd.DataFrame({"gene_idx": range(first.n_vars), "gene_name": first.var_names.astype(str)}).to_csv(
        pre_dir / "gene_index.csv", index=False
    )
    pd.DataFrame(post_qc_rows).sort_values(["metal_class", "sample_id"]).to_csv(
        pre_dir / "sample_metadata.csv", index=False
    )
    raw_metadata.to_csv(pre_dir / "sample_metadata_before_qc.csv", index=False)
    feature_table.to_csv(pre_dir / "metal_features_model_input.csv", index=False)
    save_json(
        {
            "run_name": run_name,
            "gene_selection": "seurat_v3 HVGs fitted on training perturbation samples and all controls",
            "validation_test_expression_used_for_gene_selection": False,
            "train_sample_ids": split["train_sample_ids"],
            "n_genes": int(first.n_vars),
        },
        str(pre_dir / f"gene_selection_{run_name}.json"),
    )
    merged = OmegaConf.merge(data_config, training_config)
    OmegaConf.resolve(merged)
    save_config(merged, str(pre_dir / f"config_{run_name}.yaml"))
    summary = ["# MERGE Data Summary", "", pd.DataFrame(summary_rows).to_markdown(index=False)]
    write_text(str(pre_dir / "data_summary.md"), "\n".join(summary))
    logger.info("Preprocessing completed with %d shared genes", first.n_vars)


if __name__ == "__main__":
    main()
