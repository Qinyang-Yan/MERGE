import torch

from merge.training.losses import WeightedCrossEntropyLoss


def test_weighted_ce_with_ignore_index():
    loss_fn = WeightedCrossEntropyLoss(torch.ones(3), ignore_index=-1)
    logits = torch.randn(1, 2, 3, requires_grad=True)
    labels = torch.tensor([[1, -1]])
    loss = loss_fn(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)


def test_weighted_ce_perfect_pred_zero_loss():
    loss_fn = WeightedCrossEntropyLoss(torch.ones(3), ignore_index=-1)
    logits = torch.tensor([[[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]]])
    labels = torch.tensor([[0, 1]])
    assert loss_fn(logits, labels).item() < 1e-4


def test_all_ignored_labels_return_differentiable_zero():
    loss_fn = WeightedCrossEntropyLoss(torch.ones(3), ignore_index=-1)
    logits = torch.randn(1, 2, 3, requires_grad=True)
    loss = loss_fn(logits, torch.full((1, 2), -1))
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
