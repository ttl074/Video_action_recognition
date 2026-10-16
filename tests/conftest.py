import os
import pytest


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
    categories = [d for d in os.listdir(ucf50_dataset) if os.path.isdir(os.path.join(ucf50_dataset, d))]
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
        pytest.skip(f"UCF50 dataset frames have not been extracted yet at {frames_path}. Run frame-extraction script first")
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
    categories = [d for d in os.listdir(ucf50_frames_dir) if os.path.isdir(os.path.join(ucf50_frames_dir, d))]
    video_paths = []
    for category in categories:
        category_path = os.path.join(ucf50_frames_dir, category)
        videos = os.listdir(category_path)
        video_paths.extend([os.path.join(category_path, video) for video in videos])
    return video_paths