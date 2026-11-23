"""Validation-based checkpointing and early stopping."""

from pathlib import Path

import torch
from torch import nn


class EarlyStopping:
    """Stop after a fixed number of epochs without sufficient improvement."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4) -> None:
        if patience < 1 or mode not in {"min", "max"} or min_delta < 0:
            raise ValueError("Invalid early-stopping configuration")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_epochs = 0

    def state_dict(self) -> dict:
        """Return complete early-stopping state for exact resume."""
        return {"best": self.best, "bad_epochs": int(self.bad_epochs)}

    def load_state_dict(self, state: dict) -> None:
        """Restore best score and consecutive non-improving epoch count."""
        self.best = state.get("best")
        self.bad_epochs = int(state.get("bad_epochs", 0))

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = float(metric)
            return False
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best = float(metric)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class CheckpointManager:
    """Save complete optimizer/scheduler state and the best validation model."""

    def __init__(
        self, save_dir: str, monitor: str = "val_macro_f1", mode: str = "max", save_last: bool = True
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Checkpoint mode must be 'min' or 'max'")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_last = bool(save_last)
        self.best_metric = None

    def is_best(self, metrics: dict) -> bool:
        metric = metrics.get(self.monitor)
        if metric is None:
            raise KeyError(f"Validation metrics do not contain {self.monitor}")
        if self.best_metric is None:
            self.best_metric = float(metric)
            return True
        improved = metric > self.best_metric if self.mode == "max" else metric < self.best_metric
        if improved:
            self.best_metric = float(metric)
        return bool(improved)

    def save(
        self, model, optimizer, scheduler, epoch: int, metrics: dict, is_best: bool, early_stopping=None
    ) -> None:
        state = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "early_stopping_state_dict": early_stopping.state_dict() if early_stopping is not None else None,
        }
        if self.save_last:
            torch.save(state, self.save_dir / "last.pt")
        if is_best:
            torch.save(state, self.save_dir / "best.pt")

    def load_best(self, model: nn.Module) -> nn.Module:
        state = torch.load(self.save_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=True)
        return model
