"""
Module: test_models.py

Unit tests for the LRCN model and the R3D-18 weight-inflation helpers (key_translation and
inflate_conv_weight). They use small random inputs and untrained models, so they run
quickly on CPU without the dataset or a network connection.
"""
import collections
import pytest
import torch
from torchvision.models import resnet18
from torchvision.models.video import r3d_18
from models import LRCN, LRCNConfig, key_translation, inflate_conv_weight

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


@pytest.mark.parametrize("t", [1, 3])
def test_inflate_conv_weight(t):
    """
    Test that inflate_conv_weight turns a 2D conv weight into a correctly scaled 3D one.

    Runs for t=3 (stem and 3x3x3 convs) and t=1 (downsample convs). Checks that the output
    has shape (out, in, t, k, k), that every time slice equals the 2D weight divided by t,
    that the slices sum over time back to the 2D weight, and that the input weight is left
    unchanged. Guards against a missing or wrong divide by t, which would make the 3D convs
    disagree with the copied BatchNorm running statistics, and against modifying the source
    ResNet weights in place.
    """
    torch.manual_seed(2026)
    test_weight = torch.randn(4, 2, 3, 3)
    inflated_weight = inflate_conv_weight(test_weight, t)
    shape_message = f"Expected shape {(4, 2, t, 3, 3)} got {inflated_weight.shape}"
    assert inflated_weight.shape == (4, 2, t, 3, 3),shape_message
    not_close = []
    for i in range(t):
        expected = test_weight / t
        if not torch.allclose(inflated_weight[:, :, i], expected):
            max_diff = (inflated_weight[:, :, i] - expected).abs().max().item()
            not_close.append((i, max_diff))
    not_close_message = f"Slices not close to weight / t (slice, max abs diff): {not_close}"
    assert not not_close, not_close_message


def test_key_translation():
    """
    Test that every R3D-18 weight maps onto exactly one compatible ResNet18 weight.

    Builds both models with weights=None, since key names and shapes do not depend on the
    weight values, so nothing is downloaded. Checks that every non-fc R3D key translates to
    an existing ResNet18 key with a compatible shape (equal, or equal after inflating the
    2D conv weight), that no two R3D keys translate to the same ResNet18 key, and that every
    non-fc ResNet18 weight is used. Guards against key names that do not line up, which
    would leave layers at random initialization or give one layer another layer's weights.
    """
    model_3d_state = r3d_18(weights=None).state_dict()
    model_2d_state = resnet18(weights=None).state_dict()
    translations = []
    failures = []
    for key, weight in model_3d_state.items():
        if key.startswith('fc.'):
            continue
        translation = key_translation(key)
        translations.append(translation)
        if translation not in model_2d_state:
            failures.append((key, translation, 'missing'))
            continue
        weight_2d = model_2d_state[translation]
        if weight.shape == weight_2d.shape:
            pass
        elif weight.ndim == 5 and weight_2d.ndim == 4:
            inflated_shape = inflate_conv_weight(weight_2d, weight.shape[2]).shape
            if weight.shape != inflated_shape:
                failures.append((key, translation, weight.shape, inflated_shape))
        else:
            failures.append((key, translation, weight.shape, weight_2d.shape))
    failure_message = (f"{len(failures)} keys failed "
                       "(R3D key, translated key, shape, expected shape); "
                       f"first 5: {failures[:5]}")
    assert not failures, failure_message

    counts = collections.Counter(translations)
    dups = [(key, count) for key, count in counts.items() if count > 1]
    assert not dups, f"Duplicate translations (ResNet18 key, count): {dups[:5]}"

    expected = {key for key in model_2d_state if not key.startswith('fc.')}
    translated = set(translations)
    coverage_message = (f"ResNet18 keys never mapped: {sorted(expected - translated)[:5]}; "
                        f"translated keys not in ResNet18: {sorted(translated - expected)[:5]}")
    assert translated == expected, coverage_message
