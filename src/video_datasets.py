"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using stratified sampling, and custom collate functions for handling variable-length video sequences.
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
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

def get_group(vid_path):
    """
    Extract group number from each video path
    """
    vid = os.path.basename(vid_path)
    group = vid.split('_')[-2]
    if group.startswith("g") and group[1:].isdigit():
        return int(group[1:])
    else:
        raise ValueError(f"Unexpected tag: {group} at {vid_path}")


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.
    
    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.
    
    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transform to entire clip,
        and return the stacked tensor of frames along with its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths from the video directory and select up to fr_per_vid frames
        fr_paths = list_frames(self.dataset[idx][0])
        #fr_paths = fr_paths[:self.fpv]
        
        # Open images using PIL
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        
        # Get the label associated with the video
        fr_label = self.dataset[idx][1]
        
        # Apply transforms to each frame if provided, else keep original images
        fr_imgs_trans = self.transforms(fr_imgs) if self.transforms else v2.ToImage()(fr_imgs)

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label


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


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=2026):
    """
    Split the dataset into training, validation, and test sets by group number
    
    This function splits the dataset by group number. Each category has videos belonging to the same
    group that should not be shared among splits.
    
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
    vid_paths = np.array([vid_path for vid_path in vid_dataset.keys()])
    vid_labels = np.array([vid_label for vid_label in vid_dataset.values()])
    print('Splitting train/validation/test datasets....')
    groups = np.array([get_group(vid_path) for vid_path in vid_paths])
    # UCF50 documents 25 groups per class, but 6 classes have extra groups 26-30 (63 clips).
    # Temporarily excluded pending instructor guidance; revisit to compute shared groups from data.
    keep = np.isin(groups, np.arange(1, 26))
    print(f'Dropping {np.sum(~keep)} videos in groups outside 1-25')
    vid_paths, vid_labels, groups = vid_paths[keep], vid_labels[keep], groups[keep]
    group_nums = np.unique(groups)  

    num_groups = len(group_nums)
    n_train = round(tr_ratio*num_groups)
    n_test = round(ts_ratio*num_groups)
    n_val = num_groups - n_train - n_test
    rng = np.random.default_rng(seed)
    rng.shuffle(group_nums)
    train_nums,val_nums,test_nums = group_nums[:n_train],group_nums[n_train:n_train+n_val], group_nums[n_train+n_val:]
    print(f'Train groups: {np.sort(train_nums)}')
    print(f'Validation groups: {np.sort(val_nums)}')
    print(f'Test groups: {np.sort(test_nums)}')
    train_mask = np.isin(groups,train_nums)
    tr_dataset = list(zip(vid_paths[train_mask],vid_labels[train_mask]))
    val_mask = np.isin(groups,val_nums)
    val_dataset = list(zip(vid_paths[val_mask],vid_labels[val_mask]))
    test_mask = np.isin(groups,test_nums)
    ts_dataset = list(zip(vid_paths[test_mask],vid_labels[test_mask]))
    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
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
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))
    
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    
    return padded_imgs, labels_tensor



