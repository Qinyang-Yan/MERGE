"""Optimizer and learning-rate schedule for MERGE."""

import math

import torch
from torch import nn


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Construct AdamW with separate backbone and new-module learning rates."""
    if str(config.get("type", "AdamW")).lower() != "adamw":
        raise ValueError("MERGE supports AdamW optimization")
    backbone_parameters = []
    new_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = backbone_parameters if name.startswith("gene_compass.") else new_parameters
        target.append(parameter)
    if not backbone_parameters or not new_parameters:
        raise ValueError("Both pretrained backbone and new modules require trainable parameters")
    groups = [
        {"params": backbone_parameters, "lr": float(config.get("genecompass_lr", 1e-5))},
        {"params": new_parameters, "lr": float(config.get("new_params_lr", 1e-4))},
    ]
    return torch.optim.AdamW(
        groups,
        weight_decay=float(config.get("weight_decay", 0.01)),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
    )


def build_scheduler(optimizer, config: dict, num_optimizer_steps: int):
    """Build linear warmup followed by cosine decay over optimizer updates."""
    if config.get("type", "warmup_cosine") != "warmup_cosine":
        raise ValueError("MERGE supports the warmup_cosine scheduler")
    if num_optimizer_steps <= 0:
        raise ValueError("num_optimizer_steps must be positive")
    warmup_steps = max(1, int(num_optimizer_steps * float(config.get("warmup_ratio", 0.1))))
    min_lr_ratio = float(config.get("min_lr_ratio", 0.01))
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must lie in [0, 1]")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, num_optimizer_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
