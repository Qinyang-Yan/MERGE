"""Training loop with masked validation, AMP, accumulation, and resume support."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_metrics_from_confusion
from ..utils.logging import JsonLinesWriter
from .callbacks import CheckpointManager, EarlyStopping


class Trainer:
    """Optimize MERGE and select checkpoints solely by validation performance."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        config: dict,
        logger,
        log_dir: str,
        ckpt_dir: str,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = str(device)
        self.config = config
        self.logger = logger
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.train_jsonl = JsonLinesWriter(str(self.log_dir / "train.jsonl"))
        self.eval_jsonl = JsonLinesWriter(str(self.log_dir / "eval.jsonl"))
        training = config.get("training", {})
        self.num_epochs = int(training.get("num_epochs", 50))
        self.accum_steps = int(training.get("gradient_accumulation_steps", 1))
        if self.num_epochs < 1 or self.accum_steps < 1:
            raise ValueError("num_epochs and gradient_accumulation_steps must be positive")
        self.mixed_precision = bool(training.get("mixed_precision", True)) and self.device == "cuda"
        self.grad_clip = float(training.get("gradient_clip_norm", 1.0))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        early = config.get("early_stopping", {})
        self.early_stopping_monitor = str(early.get("monitor", "val_macro_f1"))
        self.early_stopping = (
            EarlyStopping(
                patience=int(early.get("patience", 10)),
                mode=str(early.get("mode", "max")),
                min_delta=float(early.get("min_delta", 1e-4)),
            )
            if early.get("enabled", True)
            else None
        )
        checkpoint = config.get("checkpoint", {})
        self.ckpt = CheckpointManager(
            str(self.ckpt_dir),
            monitor=str(checkpoint.get("monitor", "val_macro_f1")),
            mode=str(checkpoint.get("mode", "max")),
            save_last=bool(checkpoint.get("save_last", True)),
        )
        self.ignore_index = int(getattr(loss_fn, "ignore_index", -1))
        self.global_step = 0
        self.start_epoch = 0
        self.best_metrics = {}
        self._try_resume()

    def _autocast(self):
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        return torch.autocast(
            device_type=self.device,
            dtype=dtype,
            enabled=self.mixed_precision,
        )

    def _try_resume(self) -> None:
        last = self.ckpt_dir / "last.pt"
        if not last.exists():
            return
        state = torch.load(last, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        if state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.early_stopping is not None and state.get("early_stopping_state_dict") is not None:
            self.early_stopping.load_state_dict(state["early_stopping_state_dict"])
        self.start_epoch = int(state.get("epoch", -1)) + 1
        best = self.ckpt_dir / "best.pt"
        if best.exists():
            best_state = torch.load(best, map_location="cpu", weights_only=True)
            self.best_metrics = dict(best_state.get("metrics", {}))
            best_metric = self.best_metrics.get(self.ckpt.monitor)
            if best_metric is not None:
                self.ckpt.best_metric = float(best_metric)
                if self.early_stopping is not None and self.early_stopping.best is None:
                    self.early_stopping.best = float(best_metric)
        train_log = self.log_dir / "train.jsonl"
        if train_log.exists():
            with train_log.open("r", encoding="utf-8") as handle:
                self.global_step = sum(1 for _ in handle)
        else:
            self.global_step = self.start_epoch * len(self.train_loader)
        self.logger.info("Resumed training from %s at epoch %d", last, self.start_epoch)

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            x_ctrl = batch["x_ctrl"].to(self.device)
            metal_feat = batch["metal_feat"].to(self.device)
            labels = batch["labels"].to(self.device)
            with self._autocast():
                logits = self.model(x_ctrl, metal_feat)
                loss = self.loss_fn(logits, labels) / self.accum_steps
            self.scaler.scale(loss).backward()
            if (step + 1) % self.accum_steps == 0 or step + 1 == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid = labels != self.ignore_index
                total_correct += int((predictions[valid] == labels[valid]).sum().item())
                total_count += int(valid.sum().item())
                total_loss += float(loss.item() * self.accum_steps)
            self.global_step += 1
            batch_metrics = {"train_loss": float(loss.item() * self.accum_steps)}
            self._log_metrics(batch_metrics, self.global_step, "train")
            if self.global_step % 50 == 0:
                self.logger.info("epoch=%d step=%d train_loss=%.5f", epoch, self.global_step, batch_metrics["train_loss"])
        return {
            "train_loss": total_loss / max(1, len(self.train_loader)),
            "train_acc": total_correct / max(1, total_count),
        }

    def validate(self, epoch: int) -> dict:
        self.model.eval()
        losses = []
        confusion = torch.zeros((3, 3), dtype=torch.long)
        with torch.no_grad():
            for batch in self.val_loader:
                with self._autocast():
                    logits = self.model(
                        batch["x_ctrl"].to(self.device), batch["metal_feat"].to(self.device)
                    )
                    loss = self.loss_fn(logits, batch["labels"].to(self.device))
                losses.append(float(loss.item()))
                labels = batch["labels"].to(self.device)
                predictions = logits.argmax(dim=-1)
                valid = labels != self.ignore_index
                if valid.any():
                    encoded = labels[valid].long() * 3 + predictions[valid].long()
                    confusion += torch.bincount(encoded.cpu(), minlength=9).reshape(3, 3)
        if int(confusion.sum()) == 0:
            raise ValueError("Validation contains no valid direction labels")
        metrics = compute_metrics_from_confusion(confusion.numpy())
        metrics = {
            f"val_{key}": value for key, value in metrics.items() if key != "confusion_matrix"
        }
        f1 = metrics["val_f1_per_class"]
        metrics.update(
            {
                "val_down_f1": float(f1[0]),
                "val_unchanged_f1": float(f1[1]),
                "val_up_f1": float(f1[2]),
                "val_loss": float(np.mean(losses)),
                "epoch": int(epoch),
            }
        )
        self._log_metrics(metrics, epoch, "eval")
        return metrics

    def fit(self) -> dict:
        """Train until completion or validation-based early stopping."""
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                train_metrics = self.train_one_epoch(epoch)
                validation_metrics = self.validate(epoch)
                is_best = self.ckpt.is_best(validation_metrics)
                should_stop = (
                    self.early_stopping(validation_metrics[self.early_stopping_monitor])
                    if self.early_stopping is not None
                    else False
                )
                self.ckpt.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    validation_metrics,
                    is_best=is_best,
                    early_stopping=self.early_stopping,
                )
                if is_best:
                    self.best_metrics = validation_metrics
                self.logger.info("epoch=%d metrics=%s", epoch, {**train_metrics, **validation_metrics})
                if should_stop:
                    self.logger.info("Early stopping triggered at epoch %d", epoch)
                    break
            return self.best_metrics
        finally:
            self.train_jsonl.close()
            self.eval_jsonl.close()

    def _log_metrics(self, metrics: dict, step: int, split: str) -> None:
        writer = self.train_jsonl if split == "train" else self.eval_jsonl
        writer.write({"step": int(step), **metrics})
