"""
Module: test_models.py

Unit tests for the LRCN model. They use small random inputs and an untrained backbone,
so they run quickly on CPU without the dataset or a network connection.
"""
import torch
from models import LRCN, LRCNConfig


def test_batch_independence():
    """
    Test that each clip's output does not depend on the other clips in its batch.

    Guards against the original LSTM bug, where (B, 1, F) inputs were read as
    (seq, batch, feature), so the LSTM treated the batch as a time sequence and mixed
    information across clips.
    """
    torch.manual_seed(2026)
    model = LRCN(LRCNConfig(hidden_size=32, n_layers=1, dropout_rate=0.9, n_classes=50,
                            pretrained=False, cnn_model='resnet18'))
    model.eval()
    x = torch.randn(3, 4, 3, 64, 64)
    mismatches = []
    with torch.no_grad():
        batched = model(x)
        for i, batched_row in enumerate(batched):
            single = model(x[i:i+1])[0]
            if not torch.allclose(batched_row, single, atol=1e-5):
                mismatches.append((i, (batched_row - single).abs().max().item()))
    assert not mismatches, f"Mismatches (clip, max abs diff): {mismatches[:5]}"
