#!/usr/bin/env python
"""Train MERGE with a frozen sample-level split and validation model selection."""

import argparse
import math
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from merge.data.dataset import SingleCellPerturbDataset
from merge.data.labels import compute_class_weights
from merge.data.metal_features import (
    apply_scaler,
    fit_scaler,
    get_feature_matching_mode,
    prepare_feature_table,
    save_stats,
)
from merge.data.pairing import CellTypeStratifiedPairer
from merge.data.splitting import load_sample_split
from merge.models.full_model import build_model
from merge.training.losses import build_loss
from merge.training.optimizers import build_optimizer, build_scheduler
from merge.training.trainer import Trainer
from merge.utils.config import load_config, save_config
from merge.utils.gpu import get_device
from merge.utils.io import save_json
from merge.utils.logging import setup_logger
from merge.utils.seed import seed_worker, set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_processed(data_config):
    root = Path(str(data_config.output_root)) / "processed"
    conditions = [str(data_config.ctrl_class), *map(str, data_config.metal_classes)]
    return {
        condition: ad.read_h5ad(root / f"{condition}_processed.h5ad") for condition in conditions
    }


def _load_labels(data_config, run_name):
    directory = Path(str(data_config.output_root)) / "pseudobulk" / run_name
    labels = {}
    for path in sorted(directory.glob("labels_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            labels[path.stem.removeprefix("labels_")] = data["labels"]
    if not labels:
        raise FileNotFoundError(f"No pseudo-bulk labels found in {directory}")
    return labels


def _make_datasets(adatas, metals, control, features, feature_ids, labels, celltype_to_idx, sample_ids):
    datasets = []
    selected_samples = set(map(str, sample_ids))
    for metal in metals:
        adata = adatas[str(metal)]
        subset = adata[adata.obs["sample_id"].astype(str).isin(selected_samples)].copy()
        if subset.n_obs == 0:
            continue
        subset.uns["sample_id_order"] = list(feature_ids)
        datasets.append(
            SingleCellPerturbDataset(
                subset,
                control,
                features,
                labels,
                celltype_to_idx,
                CellTypeStratifiedPairer(control),
            )
        )
    if not datasets:
        raise ValueError(f"No cells found for metals={metals} and the requested partition")
    return ConcatDataset(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs/model_config.yaml"))
    parser.add_argument("--training-config", default=str(PROJECT_ROOT / "configs/training_config.yaml"))
    parser.add_argument("--override", nargs="*", default=None, help="Training-config dot-list overrides")
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last.pt")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing completed/partial run")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config, args.override)
    run_name = str(training_config.run_name)
    seed = int(training_config.seed)
    set_global_seed(seed)
    output_root = Path(str(data_config.output_root))
    checkpoint_dir = output_root / "checkpoints" / run_name
    if args.resume:
        if not (checkpoint_dir / "last.pt").is_file():
            raise FileNotFoundError(f"Cannot resume: {checkpoint_dir / 'last.pt'} does not exist")
    elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Run directory exists: {checkpoint_dir}; use --overwrite to replace it")
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"train_{run_name}", str(output_root / "logs" / run_name / "stdout.log"))
    split = load_sample_split(str(output_root / "reports" / f"{run_name}_split.json"))
    adatas = _load_processed(data_config)
    labels = _load_labels(data_config, run_name)
    metadata_path = output_root / "preprocessing" / "sample_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Post-QC sample metadata not found: {metadata_path}")
    sample_metadata = pd.read_csv(metadata_path, dtype=str)
    sample_metadata = sample_metadata[sample_metadata["sample_id"].isin(labels)].copy()
    if sample_metadata.empty:
        raise ValueError("No post-QC sample metadata matches the pseudo-bulk labels")

    raw_features = pd.read_csv(str(data_config.metal_feat_csv))
    feature_table = prepare_feature_table(raw_features, get_feature_matching_mode(data_config))
    training_rows = sample_metadata[sample_metadata["sample_id"].isin(split["train_sample_ids"])]
    training_feature_ids = sorted(set(training_rows["metal_feature_id"].astype(str)))
    training_features = feature_table[feature_table["sample_id"].astype(str).isin(training_feature_ids)]
    if training_features.empty or set(training_feature_ids) - set(training_features["sample_id"].astype(str)):
        raise ValueError("Descriptor rows are missing for one or more training conditions")
    scaler = fit_scaler(training_features)
    save_stats(scaler, str(output_root / "preprocessing" / f"metal_feature_stats_{run_name}.json"))
    features = apply_scaler(feature_table, scaler)
    feature_ids = feature_table["sample_id"].astype(str).tolist()
    np.savez_compressed(
        output_root / "pseudobulk" / run_name / "metal_features_standardized.npz",
        features=features,
        sample_ids=np.asarray(feature_ids),
    )

    train_label_values = [
        labels[sample_id].reshape(-1)
        for sample_id in split["train_sample_ids"]
        if sample_id in labels
    ]
    if not train_label_values:
        raise ValueError("No training labels match the frozen split")
    class_weights = compute_class_weights(
        np.concatenate(train_label_values), ignore_index=int(training_config.labels.ignore_index)
    )
    save_json(
        {"class_weights": class_weights.tolist(), "source": "training labels only"},
        str(output_root / "preprocessing" / f"class_weights_{run_name}.json"),
    )

    cell_types = sorted(adatas[str(data_config.ctrl_class)].obs["cell_type"].astype(str).unique())
    celltype_to_idx = {cell_type: index for index, cell_type in enumerate(cell_types)}
    train_metals = sorted(set(training_rows["metal_class"]))
    validation_rows = sample_metadata[sample_metadata["sample_id"].isin(split["val_sample_ids"])]
    validation_metals = sorted(set(validation_rows["metal_class"]))
    train_dataset = _make_datasets(
        adatas,
        train_metals,
        adatas[str(data_config.ctrl_class)],
        features,
        feature_ids,
        labels,
        celltype_to_idx,
        split["train_sample_ids"],
    )
    validation_dataset = _make_datasets(
        adatas,
        validation_metals,
        adatas[str(data_config.ctrl_class)],
        features,
        feature_ids,
        labels,
        celltype_to_idx,
        split["val_sample_ids"],
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(training_config.training.batch_size),
        "num_workers": int(training_config.training.num_workers),
        "pin_memory": bool(training_config.training.pin_memory),
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    gene_names = adatas[str(data_config.ctrl_class)].var_names.astype(str).tolist()
    model = build_model(model_config, n_genes=len(gene_names), gene_names=gene_names)
    save_json(
        {
            "n_genes": len(gene_names),
            "mapped_genes": len(gene_names) - len(model.gene_compass.unmapped_gene_names),
            "mapping_coverage": model.gene_compass.gene_mapping_coverage,
            "unmapped_genes": model.gene_compass.unmapped_gene_names,
            "unmapped_policy": "padding token; array indices are never used as gene tokens",
        },
        str(output_root / "preprocessing" / f"gene_mapping_{run_name}.json"),
    )
    loss_config = OmegaConf.to_container(training_config.loss, resolve=True)
    loss_config["ignore_index"] = int(training_config.labels.ignore_index)
    loss_fn = build_loss(loss_config, torch.tensor(class_weights))
    optimizer = build_optimizer(model, training_config.optimizer)
    accumulation = int(training_config.training.gradient_accumulation_steps)
    optimizer_steps = math.ceil(len(train_loader) / accumulation) * int(training_config.training.num_epochs)
    scheduler = build_scheduler(optimizer, training_config.scheduler, optimizer_steps)
    merged = OmegaConf.merge(data_config, model_config, training_config)
    OmegaConf.resolve(merged)
    save_config(merged, str(checkpoint_dir / "config.yaml"))
    trainer = Trainer(
        model,
        train_loader,
        validation_loader,
        loss_fn,
        optimizer,
        scheduler,
        get_device(True),
        OmegaConf.to_container(training_config, resolve=True),
        logger,
        str(output_root / "logs" / run_name),
        str(checkpoint_dir),
    )
    best_metrics = trainer.fit()
    save_json(
        {"seed": seed, "run_name": run_name, "split": split, "best_validation_metrics": best_metrics},
        str(output_root / "reports" / f"{run_name}_training_result.json"),
    )
    logger.info("Training completed; best checkpoint: %s", checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
