import pytest
import torch

from merge.models.metal_encoder import MetalEncoder


def test_metal_encoder_output_shape():
    model = MetalEncoder()
    output = model(torch.randn(4, 5))
    assert output.shape == (4, 768)


def test_metal_encoder_gradient_flow():
    model = MetalEncoder()
    output = model(torch.randn(4, 5)).sum()
    output.backward()
    gradients = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(gradients) > 0


def test_metal_encoder_rejects_wrong_feature_count():
    model = MetalEncoder()
    with pytest.raises(ValueError):
        model(torch.randn(2, 4))
