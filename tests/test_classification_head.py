import torch

from merge.models.classification_head import DirectionClassificationHead


def test_head_output_shape():
    head = DirectionClassificationHead(hidden_dim=768, n_genes=10)
    output = head(torch.randn(3, 768))
    assert output.shape == (3, 10, 3)


def test_head_softmax_sum_to_one():
    head = DirectionClassificationHead(hidden_dim=768, n_genes=10)
    output = torch.softmax(head(torch.randn(3, 768)), dim=-1)
    assert torch.allclose(output.sum(dim=-1), torch.ones(3, 10), atol=1e-6)
