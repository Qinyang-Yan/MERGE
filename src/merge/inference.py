"""Control-cell sampling and group-level inference for new material conditions."""

import logging
from typing import Optional, Sequence

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


class MaterialControlDataset(Dataset):
    """Pair material descriptors with a fixed, seeded sample of control cells."""

    def __init__(
        self,
        ctrl_adata,
        standardized_features: np.ndarray,
        materials: Sequence[str],
        cell_types: Sequence[str],
        group_sizes: np.ndarray,
        seed: int,
    ) -> None:
        self.ctrl = ctrl_adata
        self.materials = [str(value) for value in materials]
        self.cell_types = [str(value) for value in cell_types]
        features = np.asarray(standardized_features, dtype=np.float32)
        sizes = np.asarray(group_sizes, dtype=np.int64)
        expected = (len(self.materials), len(self.cell_types))
        if features.shape != (len(self.materials), 5):
            raise ValueError(f"Expected descriptors with shape {(len(self.materials), 5)}, got {features.shape}")
        if sizes.shape != expected or (sizes < 0).any():
            raise ValueError(f"group_sizes must be non-negative with shape {expected}")
        self.features = features
        self.material_idx = []
        self.cell_type_idx = []
        for material_index in range(len(self.materials)):
            for cell_type_index in range(len(self.cell_types)):
                count = int(sizes[material_index, cell_type_index])
                self.material_idx.extend([material_index] * count)
                self.cell_type_idx.extend([cell_type_index] * count)
        self.material_idx = np.asarray(self.material_idx, dtype=np.int64)
        self.cell_type_idx = np.asarray(self.cell_type_idx, dtype=np.int64)
        control_types = ctrl_adata.obs["cell_type"].astype(str).to_numpy()
        pools = {
            cell_type: np.flatnonzero(control_types == cell_type) for cell_type in self.cell_types
        }
        empty = [cell_type for cell_type, pool in pools.items() if len(pool) == 0]
        if empty:
            raise ValueError(f"Control reference has no cells for annotated types: {empty}")
        rng = np.random.default_rng(seed)
        self.ctrl_indices = np.empty(len(self.material_idx), dtype=np.int64)
        for cell_type_index, cell_type in enumerate(self.cell_types):
            rows = np.flatnonzero(self.cell_type_idx == cell_type_index)
            if len(rows):
                self.ctrl_indices[rows] = rng.choice(
                    pools[cell_type], size=len(rows), replace=True
                )

    def __len__(self) -> int:
        return len(self.material_idx)

    def __getitem__(self, index: int) -> dict:
        row = self.ctrl.X[int(self.ctrl_indices[index])]
        values = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        material_index = int(self.material_idx[index])
        return {
            "x_ctrl": torch.as_tensor(values, dtype=torch.float32),
            "metal_feat": torch.as_tensor(self.features[material_index], dtype=torch.float32),
            "material_idx": material_index,
            "cell_type_idx": int(self.cell_type_idx[index]),
        }


def run_material_inference(
    model,
    loader,
    n_materials: int,
    n_celltypes: int,
    n_genes: int,
    device: str,
    labels: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
    ignore_index: int = -1,
) -> dict:
    """Aggregate control-cell predictions by material and cell type.

    When labels are provided, single-cell confusion matrices compare every
    sampled control-cell prediction with its group-level measured target.
    """
    if labels is not None:
        labels = np.asarray(labels)
        expected = (n_materials, n_celltypes, n_genes)
        if labels.shape != expected:
            raise ValueError(f"Expected labels with shape {expected}, got {labels.shape}")
    votes = np.zeros((n_materials, n_celltypes, n_genes, 3), dtype=np.int32)
    probability_sums = np.zeros(
        (n_materials, n_celltypes, n_genes, 3), dtype=np.float64
    )
    group_counts = np.zeros((n_materials, n_celltypes), dtype=np.int64)
    singlecell = np.zeros((3, 3), dtype=np.int64)
    by_material = np.zeros((n_materials, 3, 3), dtype=np.int64)
    by_celltype = np.zeros((n_celltypes, 3, 3), dtype=np.int64)
    model.eval()
    use_amp = str(device) == "cuda"
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            with torch.autocast(
                device_type=str(device), dtype=torch.float16, enabled=use_amp
            ):
                logits = model(batch["x_ctrl"].to(device), batch["metal_feat"].to(device))
                probabilities = torch.softmax(logits.float(), dim=-1)
            predictions = logits.argmax(dim=-1).cpu().numpy()
            probs = probabilities.cpu().numpy()
            material_indices = batch["material_idx"].numpy()
            cell_type_indices = batch["cell_type_idx"].numpy()
            for material_index, cell_type_index in sorted(
                set(zip(material_indices.tolist(), cell_type_indices.tolist()))
            ):
                keep = (material_indices == material_index) & (
                    cell_type_indices == cell_type_index
                )
                group_predictions = predictions[keep]
                group_probabilities = probs[keep]
                count = int(keep.sum())
                group_counts[material_index, cell_type_index] += count
                for class_index in range(3):
                    votes[material_index, cell_type_index, :, class_index] += (
                        group_predictions == class_index
                    ).sum(axis=0)
                probability_sums[material_index, cell_type_index] += group_probabilities.sum(
                    axis=0, dtype=np.float64
                )
                if labels is not None:
                    valid = labels[material_index, cell_type_index] != ignore_index
                    true_group = labels[material_index, cell_type_index, valid].astype(np.int64)
                    pred_group = group_predictions[:, valid].reshape(-1).astype(np.int64)
                    true_repeated = np.tile(true_group, count)
                    confusion = np.bincount(
                        true_repeated * 3 + pred_group, minlength=9
                    ).reshape(3, 3)
                    singlecell += confusion
                    by_material[material_index] += confusion
                    by_celltype[cell_type_index] += confusion
            if logger is not None and ((batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader)):
                logger.info("Inference batches: %d/%d", batch_index + 1, len(loader))
    mean_probabilities = np.zeros_like(probability_sums, dtype=np.float32)
    nonzero = group_counts > 0
    mean_probabilities[nonzero] = (
        probability_sums[nonzero] / group_counts[nonzero, None, None]
    ).astype(np.float32)
    return {
        "predictions": votes.argmax(axis=-1).astype(np.int8),
        "mean_probabilities": mean_probabilities,
        "vote_counts": votes,
        "group_prediction_counts": group_counts,
        "singlecell_confusion": singlecell if labels is not None else None,
        "singlecell_confusion_by_material": by_material if labels is not None else None,
        "singlecell_confusion_by_celltype": by_celltype if labels is not None else None,
    }


def flatten_group_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    materials: Sequence[str],
    cell_types: Sequence[str],
    ignore_index: int = -1,
) -> dict:
    """Flatten valid group/gene labels while retaining group identities."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    if labels.shape != predictions.shape or labels.shape != probabilities.shape[:3]:
        raise ValueError("Label, prediction, and probability dimensions do not match")
    true_parts, pred_parts, prob_parts = [], [], []
    material_parts, celltype_parts, gene_parts = [], [], []
    group_confusions, group_names = [], []
    for material_index, material in enumerate(materials):
        for cell_type_index, cell_type in enumerate(cell_types):
            valid = labels[material_index, cell_type_index] != ignore_index
            if not valid.any():
                continue
            true = labels[material_index, cell_type_index, valid].astype(np.int64)
            pred = predictions[material_index, cell_type_index, valid].astype(np.int64)
            true_parts.append(true)
            pred_parts.append(pred)
            prob_parts.append(probabilities[material_index, cell_type_index, valid])
            material_parts.append(np.repeat(str(material), len(true)))
            celltype_parts.append(np.repeat(str(cell_type), len(true)))
            gene_parts.append(np.flatnonzero(valid))
            group_confusions.append(
                np.bincount(true * 3 + pred, minlength=9).reshape(3, 3)
            )
            group_names.append(f"{material}|{cell_type}")
    if not true_parts:
        raise ValueError("No valid external evaluation groups")
    return {
        "y_true": np.concatenate(true_parts),
        "y_pred": np.concatenate(pred_parts),
        "y_prob": np.concatenate(prob_parts),
        "material": np.concatenate(material_parts),
        "cell_type": np.concatenate(celltype_parts),
        "gene_idx": np.concatenate(gene_parts),
        "group_confusions": np.stack(group_confusions),
        "group_names": np.asarray(group_names),
    }
