"""
Module: test_dataset.py

Integrity tests for the UCF50 data on disk. They check that the raw dataset has 50
class folders and 6,681 .avi files, and that frame extraction produced one folder per
video, each with exactly n_frames readable JPEG frames.

These tests need the real dataset and skip automatically when it is not present
(see conftest.py for the paths and environment variables).
"""
import os
import random
import collections
from PIL import Image
import video_datasets as vd

def test_raw_class_folders(ucf50_dataset):
    """
    Test the preparation of the UCF50 dataset.
    """
    n_categories = len(os.listdir(ucf50_dataset))
    size_message = f"UCF50 dataset directory has {n_categories} categories, expected 50."
    assert n_categories == 50, size_message
    with os.scandir(ucf50_dataset) as entries:
        # Check if every single entry is a directory
        all_folders = all(entry.is_dir() for entry in entries)
    assert all_folders, "Found entities that are not folders in this directory."


def test_video_avis(video_avis):
    """
    Test that we have the correct number of video avis in the UCF50 dataset and that all are avis.
    """
    num_avi_message = f"Expected 6681 video avis in the UCF50 dataset but found {len(video_avis)}."
    assert len(video_avis) == 6681, num_avi_message
    non_avis = []
    for video_avi in video_avis:
        if not video_avi.lower().endswith(".avi"):
            non_avis.append(video_avi)
    assert not non_avis, f"{len(non_avis)} entries lack .avi extension, e.g. {non_avis[:5]}"


def test_ucf50_frames_dir(ucf50_frames_dir):
    """
    Test that there are 50 video directories in ucf50_frames
    """
    not_dirs = []
    categories = os.listdir(ucf50_frames_dir)
    for category in categories:
        if not os.path.isdir(os.path.join(ucf50_frames_dir,category)):
            not_dirs.append(category)
    assert len(categories) == 50, f"Expected 50 categories and found {len(categories)}"
    assert not not_dirs, f"Some categories are not directories: {not_dirs[:5]}"


def test_video_dirs(video_dirs):
    """
    Test that there are 6681 entry directories
    """
    dir_count = 0
    not_dirs = []
    for video_dir in video_dirs:
        if not os.path.isdir(video_dir):
            not_dirs.append(video_dir)
        else:
            dir_count += 1
    assert dir_count == 6681, f"Expected 6681 video directories, found {dir_count}"
    assert not not_dirs, f"Entries that are not directories: {not_dirs[:5]}"


def test_frame_counts(video_dirs,n_frames):
    """
    Test that there are no empty video folders in the UCF50 dataset,that all directories
    have the same number of frames, and that it equals n_frames.
    """
    empty_dirs = []
    frame_counts = {}
    for video_dir in video_dirs:
        frame_files = vd.list_frames(video_dir)
        frame_counts[video_dir] = len(frame_files)
        if len(frame_files) == 0:
            empty_dirs.append(video_dir)
    most_common = collections.Counter(frame_counts.values()).most_common(1)[0][0]
    outliers = {k:v for k,v in frame_counts.items() if v!=most_common}
    assert not empty_dirs, f"Found {len(empty_dirs)} empty video folders: {empty_dirs[:5]}"
    assert not outliers, (
        f"{len(outliers)} directories do not have the expected number of frames ({most_common}):\n"
        f"{list(outliers.items())[:5]}"
    )
    assert most_common == n_frames, f"expected {n_frames} but got {most_common}"

def test_frames_read(video_dirs):
    """
    Test that frames can be read from a random sample of video folders in the UCF50 dataset.
    """
    run_seed = 2026
    sample_size = 50
    rng = random.Random(run_seed)
    sample_video_dirs = rng.sample(video_dirs, sample_size)
    failures = []
    for sample in sample_video_dirs:
        frames = vd.list_frames(sample)
        if not frames:
            continue
        try:
            with Image.open(rng.choice(frames)) as img:
                img.verify()
        except (OSError, SyntaxError, ValueError) as e:
            failures.append((sample, e))
    assert not failures, (
        f"Failed to read frames from {len(failures)} of {sample_size} sampled video folders: "
        f"{failures[:5]}"
    )
