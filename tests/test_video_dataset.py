"""
Module: test_video_dataset.py

Tests for VideoDataset and the dataset split:
    - test_train_transform_frame_consistency: random training augmentation is drawn once
      per clip, so every frame of a clip gets the same flip/shift (runs anywhere, using a
      synthetic clip).
    - test_data_leaks: the group-aware split never puts clips from the same group in more
      than one split (needs the extracted UCF50 frames; skips otherwise).
"""
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
    dataset = vd.VideoDataset([(synthetic_clip_dir, 0)], n_frames, tr_transforms)
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
