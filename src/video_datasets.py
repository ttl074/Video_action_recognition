"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, a group-aware
split that keeps all clips from the same UCF50 group (source video) in a single split, and
custom collate functions for handling variable-length video sequences.
"""

import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
import numpy as np


def list_frames(folder):
    """
    List all JPEG frame files in the specified folder, sorted by frame number.

    Args:
        folder (str): Path to the folder containing frame images.

    Returns:
        list: List of frame file paths, sorted by frame number.
    """
    frame_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
    sorted_files = sorted(frame_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    return sorted_files


def sample_frame_indices(n_frames, fr_per_vid, train, offset=0.5):
    """
    Select fr_per_vid frame indices from a video, one per equal-length temporal segment.

    The frames are divided into fr_per_vid contiguous segments covering the whole video.
    In training mode, one frame is drawn uniformly at random from each segment (temporal
    jitter), so each epoch sees a different subset of every video. In evaluation mode,
    the frame at fractional position offset within each segment is taken (the center by
    default).

    Args:
        n_frames (int): Number of frames in the video.
        fr_per_vid (int): Number of frames to select.
        train (bool): If True, sample randomly within each segment; if False,
            take the frame at offset within each segment.
        offset (float): The fractional position within each segment for evaluation
            (0.5 = center); must be in [0, 1) to stay inside the video.

    Returns:
        list[int]: fr_per_vid frame indices, in temporal order.
    """
    if train:
        random_idx = []
        for i in range(fr_per_vid):
            start = i * n_frames // fr_per_vid
            end = (i + 1) * n_frames // fr_per_vid
            end = max(end, start + 1)
            random_idx.append(random.randint(start, end - 1))
        return random_idx
    return ((np.arange(fr_per_vid) + offset) * (n_frames / fr_per_vid)).astype(int).tolist()


def sample_frame_paths(fr_paths, fr_per_vid, train, offset=0.5):
    """
    Select fr_per_vid frame paths from a video, one per equal-length temporal segment.

    Thin wrapper around sample_frame_indices that returns the paths at those indices.

    Args:
        fr_paths (list): Frame file paths for one video, sorted by frame number.
        fr_per_vid (int): Number of frames to select.
        train (bool): If True, sample randomly within each segment; if False,
            take the frame at offset within each segment.
        offset (float): The fractional position within each segment for evaluation
            (0.5 = center); must be in [0, 1) to stay inside the video.

    Returns:
        list: fr_per_vid frame paths, in temporal order.
    """
    return [fr_paths[i] for i in sample_frame_indices(len(fr_paths), fr_per_vid, train, offset)]


def get_group(vid_path):
    """
    Extract the UCF50 group number from a video path.

    Video folders are named v_<Class>_g<NN>_c<NN>; the group is the second-to-last
    underscore-separated field.

    Args:
        vid_path (str): Path to a video folder (or a bare folder name).

    Returns:
        int: The group number (e.g. 'g07' -> 7).

    Raises:
        ValueError: If the name does not contain a group tag in the expected position.
    """
    vid = os.path.basename(vid_path)
    group = vid.split('_')[-2]
    if not (group.startswith("g") and group[1:].isdigit()):
        raise ValueError(f"Unexpected tag: {group} at {vid_path}")
    return int(group[1:])


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a list of (video_directory_path, label) pairs.

    Args:
        vid_dataset (list): List of (video directory path, integer label) tuples.
        fr_per_vid (int): Number of frames sampled from each video folder per clip, one per
            equal-length temporal segment (see sample_frame_paths).
        train (bool): If True, frames are sampled at random positions within each segment
            (temporal jitter for training); if False, segment centers are used so validation
            and test clips are deterministic.
        transforms (callable, optional): A clip-level torchvision.transforms.v2 pipeline that
            is called once on the list of PIL frames, so random augmentations are applied
            identically to every frame in the clip. If None, frames are only converted to
            uint8 tensors.
        offset (float): Fractional position within each segment used in evaluation mode
            (0.5 = center; default). Must be in [0, 1). Ignored when train is True.
    """
    def __init__(self, vid_dataset, fr_per_vid, train, transforms=None, offset=0.5):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.train = train
        self.transforms = transforms
        self.offset = offset

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def _load_clip(self, fr_paths):
        """
        Open the given frames and apply the clip-level transforms to all of them at once.

        Args:
            fr_paths (list): Frame file paths, in the order they should appear in the clip.

        Returns:
            torch.Tensor: Tensor of shape (len(fr_paths), C, H, W), or an empty list if
                fr_paths is empty.
        """
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        # Apply the clip-level transforms once to all frames, else just convert to tensors
        fr_imgs_trans = self.transforms(fr_imgs) if self.transforms else v2.ToImage()(fr_imgs)
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)
        return fr_imgs_trans

    def __getitem__(self, idx):
        """
        Load the sampled frames of the video at the given index, apply the transforms to the
        entire clip, and return the stacked tensor of frames along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T equal to fr_per_vid, and label is an integer.
        """
        fr_paths = sample_frame_paths(list_frames(self.dataset[idx][0]), self.fpv,
                                      self.train, self.offset)
        return self._load_clip(fr_paths), self.dataset[idx][1]


class DiffVideoDataset(VideoDataset):  # pylint: disable=too-few-public-methods
    """
    VideoDataset variant that returns RGB differences between adjacent frames.

    Frame positions are sampled exactly as in VideoDataset. For each sampled frame i, the
    next frame i + 1 is also loaded (the last frame is paired with itself, giving a zero
    difference), and the clip returned is frame(i + 1) - frame(i) for every sampled i. Static
    content cancels to about zero, so the clip mainly shows where things moved.

    All 2 * fr_per_vid frames go through the transforms in a single call, so the current and
    next frames receive identical crops, flips, and color jitter before being subtracted.
    Normalize is linear, so the mean cancels and differences are simply scaled by 1/std.

    Takes the same arguments as VideoDataset.
    """
    def __getitem__(self, idx):
        """
        Load the sampled frames and their next frames, and return their differences.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (diff_tensor, label) where diff_tensor is a float tensor of shape
                (T, C, H, W) with T equal to fr_per_vid, and label is an integer.
        """
        fr_paths = list_frames(self.dataset[idx][0])
        cur_idx = sample_frame_indices(len(fr_paths), self.fpv, self.train, self.offset)
        next_idx = [min(i + 1, len(fr_paths) - 1) for i in cur_idx]
        clip = self._load_clip([fr_paths[i] for i in cur_idx + next_idx]).float()
        n_sampled = len(cur_idx)
        return clip[n_sampled:] - clip[:n_sampled], self.dataset[idx][1]


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.

    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.

    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def _split_group_numbers(group_nums, tr_ratio, ts_ratio, seed):
    """
    Shuffle group numbers with a seeded RNG and divide them into train/val/test.

    Train and test sizes are rounded from their ratios; validation gets the remainder.

    Args:
        group_nums (np.ndarray): Unique group numbers to divide.
        tr_ratio (float): Proportion of the groups to use for training.
        ts_ratio (float): Proportion of the groups to use for testing.
        seed (int): Random seed for the shuffle.

    Returns:
        tuple: (train_nums, val_nums, test_nums), each an np.ndarray of group numbers.
    """
    shuffled = np.random.default_rng(seed).permutation(group_nums)
    n_train = round(tr_ratio * len(shuffled))
    n_test = round(ts_ratio * len(shuffled))
    n_val = len(shuffled) - n_train - n_test
    return (shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:])


def _subset(vid_paths, vid_labels, groups, nums):
    """
    Select the (path, label) pairs whose group number is in nums.

    Args:
        vid_paths (np.ndarray): Video paths.
        vid_labels (np.ndarray): Labels aligned with vid_paths.
        groups (np.ndarray): Group numbers aligned with vid_paths.
        nums (np.ndarray): Group numbers to keep.

    Returns:
        list: List of (video_path, label) tuples.
    """
    mask = np.isin(groups, nums)
    return list(zip(vid_paths[mask], vid_labels[mask]))


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=2026):
    """
    Split the dataset into training, validation, and test sets by group number.

    UCF50 clips within a group are cut from the same long video (same actor, background,
    and camera), so sharing a group across splits would leak information. Every clip of a
    given group number therefore goes to exactly one split, and the same group numbers are
    held out for every class, which keeps the splits balanced across classes.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the groups to use for training.
        ts_ratio (float): Proportion of the groups to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 2026.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    vid_paths = np.array(list(vid_dataset.keys()))
    vid_labels = np.array(list(vid_dataset.values()))
    print('Splitting train/validation/test datasets....')
    groups = np.array([get_group(vid_path) for vid_path in vid_paths])

    # UCF50 documents 25 groups per class, but 6 classes have extra groups 26-30 (63 clips).
    # Temporarily excluded pending instructor guidance; revisit to compute shared groups from data.
    keep = np.isin(groups, np.arange(1, 26))
    print(f'Dropping {np.sum(~keep)} videos in groups outside 1-25')
    vid_paths, vid_labels, groups = vid_paths[keep], vid_labels[keep], groups[keep]

    train_nums, val_nums, test_nums = _split_group_numbers(np.unique(groups),
                                                           tr_ratio, ts_ratio, seed)
    print(f'Train groups: {np.sort(train_nums)}')
    print(f'Validation groups: {np.sort(val_nums)}')
    print(f'Test groups: {np.sort(test_nums)}')

    return (_subset(vid_paths, vid_labels, groups, train_nums),
            _subset(vid_paths, vid_labels, groups, val_nums),
            _subset(vid_paths, vid_labels, groups, test_nums))


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).

    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames (keeping each label paired with its clip), stacks the video frame tensors,
    transposes them to (B, C, T, H, W), and builds a label tensor.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (imgs_tensor, labels_tensor), or (None, None) if every clip is empty.
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Filter clips and labels together so they stay paired
    valid_samples = [(imgs, label) for imgs, label in batch if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.tensor(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.

    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (padded_imgs, labels_tensor), or (None, None) if every clip is empty.
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Filter out any samples that have no frames, keeping clips and labels paired
    valid_samples = [(imgs, label) for imgs, label in batch if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    return padded_imgs, labels_tensor
