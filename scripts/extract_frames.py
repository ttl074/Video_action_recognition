"""
Module: extract_frames.py

This module provides functions to extract video directories from extracted dataset folder with avi files

Functions:
    - convert_to_dirs: Converts avi files in raw directory to video folders
"""
import argparse
import os
import shutil
from tqdm import tqdm
from utils import get_frames, store_frames, FrameExtractionError
from video_datasets import list_frames


def convert_to_dirs(raw_dir,n_frames,frame_dir):
    """
    Convert avi files in raw video directory to video folders
    
    Each avi file has n_frames extracted .

    Args:
        raw_dir: path to extracted video directory with avis
        n_frames: number of frames to extract from each avi
        frame_dir: directory to store video directories

    Returns:
        None

    Raises:
        FrameExtractionError: If frame extraction fails
        OSError: If frame writing fails
    
    Notes:
        Outputs to store_dir/<category>/<video name>/00000.jpg ... 1 video folder per avi
        Uses tqdm to track progress
    """
    jobs = []
    skipped = 0
    extracted = 0
    for category in os.listdir(raw_dir):
        category_path = os.path.join(raw_dir,category)
        if not os.path.isdir(category_path):
            continue
        for avi in os.listdir(category_path):
            jobs.append((category,avi))
    for category,avi in tqdm(jobs, desc="Converting from avi",unit="file"):
        category_path = os.path.join(raw_dir,category)
        video_name = os.path.splitext(avi)[0]
        avi_path = os.path.join(category_path,avi)
        store_path = os.path.join(frame_dir,category,video_name)
        if os.path.isdir(store_path) and len(list_frames(store_path)) == n_frames:
            skipped += 1
            continue
        elif os.path.isdir(store_path):
            shutil.rmtree(store_path)
        try:
            frames,_ = get_frames(avi_path,n_frames)
            store_frames(frames,store_path)
            extracted += 1
        except(FrameExtractionError, OSError):
            shutil.rmtree(store_path,ignore_errors=True)
            raise
    print(f"Extracted {extracted} and skipped {skipped} out of {len(jobs)} avis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset avis to video directories")
    parser.add_argument('--raw_dir', type=str, default='/content/datasets/UCF50', help='Path to extracted dataset folder with avis')
    parser.add_argument('--n_frames',type=int,default=16,help='Number of frames to extract from each avi')
    parser.add_argument('--frame_dir',type=str,default='/content/datasets/UCF50_frames_16',help="Directory to output video directories to")
    args = parser.parse_args()
    raw_dir = args.raw_dir
    n_frames = args.n_frames
    frame_dir = args.frame_dir
    convert_to_dirs(raw_dir,n_frames,frame_dir)
