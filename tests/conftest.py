"""
Shared pytest fixtures for the UCF50 test suite.

Provides paths to the raw dataset and the extracted frames (overridable with the
UCF50_DIR, UCF50_FRAMES_DIR, and UCF50_N_FRAMES environment variables), plus lists of
the video files and frame folders. Fixtures that need data skip their tests when the
data is not present.
"""
# Fixtures that depend on other fixtures take them as same-named parameters by design.
# pylint: disable=redefined-outer-name
import os
import shutil
import numpy as np
import pytest
from PIL import Image

@pytest.fixture(scope="session")
def n_frames():
    """Number of frames to extract from each video (env: UCF50_N_FRAMES, default=16)."""
    return int(os.environ.get("UCF50_N_FRAMES","16"))


@pytest.fixture(scope="session")
def ucf50_dataset():
    """
    Fixture to provide the path to the UCF50 dataset for testing.

    Returns:
        str: Path to the UCF50 dataset directory.
    """
    dataset_path = os.environ.get("UCF50_DIR",'/content/datasets/UCF50')
    if not os.path.isdir(dataset_path):
        pytest.skip(f"UCF50 dataset path does not exist at {dataset_path} or is not a directory.")
    return dataset_path


@pytest.fixture(scope="session")
def video_avis(ucf50_dataset):
    """
    Fixture to provide a list of video avis in the UCF50 dataset.
    
    Args:
        ucf50_dataset (str): Path to the UCF50 dataset directory.

    Returns:
        list: List of video avis in the UCF50 dataset.
    """
    categories = [d for d in os.listdir(ucf50_dataset) if
                  os.path.isdir(os.path.join(ucf50_dataset, d))]
    video_paths = []
    for category in categories:
        category_path = os.path.join(ucf50_dataset, category)
        videos = os.listdir(category_path)
        video_paths.extend([os.path.join(category_path, video) for video in videos])
    return video_paths


@pytest.fixture(scope="session")
def ucf50_frames_dir():
    """
    Fixture to provide the path to the UCF50 dataset frames for testing.

    Returns:
        str: Path to the UCF50 dataset frames directory.
    """
    frames_path = os.environ.get("UCF50_FRAMES_DIR",'/content/datasets/UCF50_frames')
    if not os.path.isdir(frames_path):
        pytest.skip(f"UCF50 dataset frames have not been extracted yet at {frames_path}.")
    return frames_path


@pytest.fixture(scope="session")
def video_dirs(ucf50_frames_dir):
    """
    Fixture to provide a list of video dirs in the UCF50 dataset.
    
    Args:
        ucf50_frames_dir (str): Path to the UCF50 frames directory.

    Returns:
        list: List of video directories in the UCF50 dataset.
    """
    categories = [d for d in os.listdir(ucf50_frames_dir)
                  if os.path.isdir(os.path.join(ucf50_frames_dir, d))]
    video_paths = []
    for category in categories:
        category_path = os.path.join(ucf50_frames_dir, category)
        videos = os.listdir(category_path)
        video_paths.extend([os.path.join(category_path, video) for video in videos])
    return video_paths


@pytest.fixture
def synthetic_clip_dir(tmp_path, n_frames):
    """
    Create a fake clip: n_frames byte-identical random-noise JPEGs named 00000.jpg, ...

    Returns:
        str: Path to the folder containing the frames.
    """
    arr = np.random.default_rng(2026).integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "frame.jpg")
    frames_dir = os.path.join(tmp_path, "frames")
    os.makedirs(frames_dir)
    for i in range(n_frames):
        shutil.copy(os.path.join(tmp_path, "frame.jpg"),
                    os.path.join(frames_dir, f"{i:05d}.jpg"))
    return frames_dir
