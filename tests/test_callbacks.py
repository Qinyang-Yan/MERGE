import logging

import numpy as np
import torch
from torch import nn

from merge.training.callbacks import EarlyStopping
from merge.training.losses import WeightedCrossEntropyLoss
from merge.training.trainer import Trainer


def test_early_stopping_state_roundtrip():
    early = EarlyStopping(patience=3, mode="max", min_delta=0.0)
    early(0.5)
    early(0.4)
    early(0.3)
    state = early.state_dict()
    restored = EarlyStopping(patience=3, mode="max", min_delta=0.0)
    restored.load_state_dict(state)
    assert restored.best == 0.5
    assert restored.bad_epochs == 2
    assert restored(0.2) is True


def test_trainer_uses_independent_early_stopping_monitor(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {
        "training": {
            "num_epochs": 3,
            "gradient_accumulation_steps": 1,
            "mixed_precision": False,
        },
        "early_stopping": {
            "enabled": True,
            "patience": 1,
            "monitor": "val_loss",
            "mode": "min",
            "min_delta": 0.0,
        },
        "checkpoint": {"monitor": "val_macro_f1", "mode": "max", "save_last": True},
    }
    trainer = Trainer(
        model,
        train_loader=[],
        val_loader=[],
        loss_fn=WeightedCrossEntropyLoss(torch.ones(3)),
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        config=config,
        logger=logging.getLogger("test"),
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "checkpoints"),
    )
    epochs = []
    validation = [
        {"val_loss": 1.0, "val_macro_f1": 0.5},
        {"val_loss": 0.9, "val_macro_f1": 0.4},
        {"val_loss": 0.8, "val_macro_f1": 0.3},
    ]
    trainer.train_one_epoch = lambda epoch: {}
    trainer.validate = lambda epoch: (epochs.append(epoch) or validation[epoch])
    trainer.fit()
    assert epochs == [0, 1, 2]
    state = torch.load(tmp_path / "checkpoints" / "last.pt", weights_only=True)
    assert state["early_stopping_state_dict"]["best"] == np.float64(0.8)

    resumed_model = nn.Linear(2, 2)
    resumed = Trainer(
        resumed_model,
        train_loader=[],
        val_loader=[],
        loss_fn=WeightedCrossEntropyLoss(torch.ones(3)),
        optimizer=torch.optim.AdamW(resumed_model.parameters(), lr=1e-3),
        scheduler=None,
        device="cpu",
        config=config,
        logger=logging.getLogger("test_resume"),
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "checkpoints"),
    )
    assert resumed.start_epoch == 3
    assert resumed.early_stopping.best == np.float64(0.8)
    resumed.train_jsonl.close()
    resumed.eval_jsonl.close()
