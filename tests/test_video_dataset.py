"""
Module: test_video_dataset.py

Tests for VideoDataset and the dataset split:
    - test_train_transform_frame_consistency: random training augmentation is drawn once
      per clip, so every frame of a clip gets the same flip/shift (runs anywhere, using a
      synthetic clip).
    - test_sample_frame_paths: segment-based frame sampling returns fr_per_vid in-order
      paths, one from each temporal segment, deterministic in eval mode and varying in
      train mode, including for videos shorter than fr_per_vid (runs anywhere, using
      fake paths).
    - test_data_leaks: the group-aware split never puts clips from the same group in more
      than one split (needs the extracted UCF50 frames; skips otherwise).
"""
import random
import torch
from utils import transform_stats, compose_data_transforms
import video_datasets as vd


def test_train_transform_frame_consistency(synthetic_clip_dir, n_frames):
    """
    Test that train transforms apply the same random parameters to every frame of a clip.

    Guards against the original bug, where transforms were applied per frame so each frame
    got its own random flip/shift.
    """
    torch.manual_seed(2026)
    h, w, mean, std = transform_stats("lrcn")
    tr_transforms, _ = compose_data_transforms(h, w, mean, std)
    dataset = vd.VideoDataset([(synthetic_clip_dir, 0)], n_frames, True, tr_transforms)
    clips = []
    failures = []
    for i in range(10):
        clip, label = dataset[0]
        if clip.shape != (n_frames, 3, h, w):
            failures.append(f"iter {i}: shape {tuple(clip.shape)}, expected {(n_frames, 3, h, w)}")
        if clip.dtype != torch.float32:
            failures.append(f"iter {i}: dtype {clip.dtype}, expected torch.float32")
        if label != 0:
            failures.append(f"iter {i}: label {label}, expected 0")
        for j in range(1, n_frames):
            if not torch.equal(clip[0], clip[j]):
                failures.append(f"iter {i}: frame {j} differs from frame 0")
        clips.append(clip)
    assert not failures, f"{len(failures)} check(s) failed, e.g. {failures[:5]}"
    assert any(not torch.equal(clips[0], c) for c in clips[1:]), (
        "all 10 clips were identical; random augmentation may not be running"
    )


def _check_sample(sampled, n_paths, fr_per_vid, case):
    """
    Check one output of sample_frame_paths and return a list of failure messages.

    Frame paths are fake names "00000.jpg", "00001.jpg", ..., so the frame index is the
    number in the name.

    Args:
        sampled (list): Paths returned by sample_frame_paths.
        n_paths (int): Number of paths that were passed in.
        fr_per_vid (int): Number of frames that were requested.
        case (str): Label for this call, used in failure messages.

    Returns:
        list: Failure messages (empty if every check passed).
    """
    failures = []
    if len(sampled) != fr_per_vid:
        return [f"{case}: got {len(sampled)} paths, expected {fr_per_vid}"]
    idx = [int(p.split('.')[0]) for p in sampled]
    if any(i < 0 or i >= n_paths for i in idx):
        failures.append(f"{case}: index out of range in {idx}")
    if idx != sorted(idx):
        failures.append(f"{case}: frames not in temporal order: {idx}")
    # With at least one frame per segment, pick i must fall inside segment i
    if n_paths >= fr_per_vid:
        for i, frame in enumerate(idx):
            start = i * n_paths // fr_per_vid
            end = (i + 1) * n_paths // fr_per_vid
            if not start <= frame < end:
                failures.append(f"{case}: pick {i} is frame {frame}, "
                                f"outside segment [{start}, {end})")
    return failures


def test_sample_frame_paths():
    """
    Test that sample_frame_paths returns fr_per_vid in-order frame paths, one from inside
    each temporal segment, for videos shorter than, equal to, and longer than fr_per_vid.
    Eval mode must return the same frames every call; train mode must vary on a long video.
    """
    random.seed(2026)
    fr_per_vid = 16
    failures = []
    for n_paths in (10, 16, 32, 181, 900):
        paths = [f"{i:05d}.jpg" for i in range(n_paths)]
        for train in (True, False):
            for _ in range(50):
                sampled = vd.sample_frame_paths(paths, fr_per_vid, train)
                failures += _check_sample(sampled, n_paths, fr_per_vid,
                                          f"n_paths={n_paths}, train={train}")
    assert not failures, f"{len(failures)} check(s) failed, e.g. {failures[:5]}"

    paths = [f"{i:05d}.jpg" for i in range(181)]
    eval_draws = {tuple(vd.sample_frame_paths(paths, fr_per_vid, False)) for _ in range(20)}
    assert len(eval_draws) == 1, "eval sampling returned different frames across calls"
    train_draws = {tuple(vd.sample_frame_paths(paths, fr_per_vid, True)) for _ in range(20)}
    assert len(train_draws) > 1, "train sampling returned the same frames every call"


def test_data_leaks(ucf50_frames_dir):
    """
    Test that the splits share no groups, none is empty, and together they cover all groups.
    """
    vid_dataset, _ = vd.load_dataset(ucf50_frames_dir)
    splits = vd.dataset_split(vid_dataset, 0.75, 0.15)
    train_g, val_g, test_g = [{vd.get_group(path) for path, _ in split} for split in splits]
    assert train_g, "Training set is empty!"
    assert val_g, "Validation set is empty!"
    assert test_g, "Test set is empty!"
    assert not train_g & val_g, f"Training and val share groups {sorted(train_g & val_g)}"
    assert not val_g & test_g, f"Val and test share groups {sorted(val_g & test_g)}"
    assert not train_g & test_g, f"Training and test share groups {sorted(train_g & test_g)}"
    num_groups = len(train_g) + len(val_g) + len(test_g)
    assert num_groups == 25, f"UCF50 has 25 groups, found {num_groups}"