"""Configuration loading and saving utilities."""

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str, overrides: Optional[list] = None) -> DictConfig:
    """Load a YAML config and apply dot-list overrides.

    Args:
        config_path: Main YAML config path.
        overrides: Optional list such as
            ``["training.batch_size=32", "model.dropout=0.2"]``.

    Returns:
        A resolved OmegaConf configuration object.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, save_path: str) -> None:
    """Save the final runtime config for reproducibility.

    Args:
        cfg: Resolved OmegaConf config.
        save_path: Destination YAML path.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path)
