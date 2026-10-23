"""
Module: utils.py

This module provides helper functions for video processing and data transformations
for video classification tasks. It includes functions for:
    - Sampling all frames from a video.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn


def sample_frames(v_cap):
    """
    Helper function to sample all frames from an open video capture. Reads all available frames.

    Args:
        v_cap (cv2.VideoCapture): An opened video capture positioned at the first frame.

    Returns:
        frames (list): List of sampled frames (as numpy arrays in RGB format).
    """
    frames = []
    success= True
    while success:
        success, frame = v_cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


class FrameExtractionError(RuntimeError):
    """Custom exception for errors during video processing."""
    def __init__(self, video,reason,retrieved_frames):
        self.video = video
        self.reason = reason
        self.retrieved_frames = retrieved_frames
        super().__init__(f"{video} got only {retrieved_frames} because {reason}")


def get_frames(vid, n_frames=16):
    """
    Sample all frames from avi files

    Args:
        vid (str): Path to the video file.
        n_frames (int): Minimum number of frames in directory
    Returns:
        tuple: (frames, true_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - true_len (int): Number of decoded frames before padding

    Raises:
        FrameExtractionError: If video cannot be opened or contains no frames
    Notes:
        If true_len < n_frames, pad (n_frames-true_len) to frames list
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    try:
        # if video cannot be opened raise error
        if not v_cap.isOpened():
            raise FrameExtractionError(vid,"failed to open video",len(frames))
        frames = sample_frames(v_cap)
        true_len = len(frames)
        # if video does not have frames raise error
        if true_len == 0:
            raise FrameExtractionError(vid,"no frames found in video",len(frames))
        # if number of extracted frames is less than target,
        # pad n_frames - true_len repeated frame at segment centers
        if true_len < n_frames:
            num_pad = n_frames - true_len
            frame_idx = ((np.arange(num_pad)+0.5)*(true_len/num_pad)).astype(int)
            idx = sorted(np.concatenate((range(true_len),frame_idx)))
            frames = [frames[i] for i in idx]
    finally:
        v_cap.release()
    return frames, true_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None

    Raises:
        OSError: If a frame failed to write

    Notes:
        Frames are saved as 00000.jpg, 00001.jpg,... (zero-padded so they sort correctly
        and match list_frames). 00000.jpg is written last to keep track of folder completion
    """
    os.makedirs(store_path,exist_ok=True)
    for idx, frame in enumerate(frames[1:],start=1):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, f"{idx:05d}.jpg")
        if not cv2.imwrite(path_to_frame, frame):
            raise OSError(f"Failed to write to {path_to_frame}")
    #write 00000.jpg last for tracking when folder write is finished
    f = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    ptf = os.path.join(store_path, "00000.jpg")
    if not cv2.imwrite(ptf, f):
        raise OSError(f"Failed to write to {ptf}")


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    Returns the output frame size and the per-channel normalization statistics for the
    given model type: 224x224 with ImageNet statistics for 'lrcn', and 112x112 with
    ImageNet statistics for '3dcnn'. The size is the final crop size produced by
    compose_data_transforms. ImageNet statistics because the R3D weights are inflated
    from ImageNet ResNet18, and 112×112 to keep the 3D network’s compute manageable on a T4.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return clip-level data transforms for training and validation/test datasets.

    Both pipelines use torchvision.transforms.v2 and are called once per clip on the list
    of PIL frames, so any random parameters (crop, flip, color jitter) are sampled once and
    applied identically to every frame in the clip. The training pipeline takes a random
    crop covering 50-100% of the frame area and resizes it to height x width, applies a
    random horizontal flip, and jitters brightness, contrast, saturation, and hue. The
    validation/test pipeline resizes the short side to height (keeping the aspect ratio) and
    center-crops to height x width, so evaluation frames are undistorted like the training
    crops. Both pipelines then convert to a float tensor in [0, 1] and normalize.

    Args:
        height (int): Output image height.
        width (int): Output image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed v2 transforms for the training set.
            - val_test_transforms: Composed v2 transforms for the validation/test set.
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((height,width), scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize(height),
        transforms.CenterCrop((height,width)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn',num_workers=0):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Number of workers to prepare batches while GPU is running
    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    pw = num_workers > 0
    if model == "lrcn":
        train_dl = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,
                            collate_fn=collate_fn_rnn,num_workers=num_workers,
                            persistent_workers=pw, pin_memory=True)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False,
                            collate_fn=collate_fn_rnn,num_workers=num_workers,
                            persistent_workers=pw, pin_memory=True)
    else:
        train_dl = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,
                            collate_fn=collate_fn_r3d_18,num_workers=num_workers,
                            persistent_workers=pw, pin_memory=True)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False,
                            collate_fn=collate_fn_r3d_18,num_workers=num_workers,
                            persistent_workers=pw, pin_memory=True)
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn',num_workers=0):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Number of workers to prepare batches while GPU is running
    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    pw = num_workers > 0

    if model == "lrcn":
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False,
                            collate_fn=collate_fn_rnn,num_workers=num_workers,
                            persistent_workers=pw, pin_memory=True)
    else:
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False,
                            collate_fn=collate_fn_r3d_18,num_workers=num_workers,
                            persistent_workers=pw, pin_memory=True)
    dataloaders = {'test': test_dl}
    return dataloaders
