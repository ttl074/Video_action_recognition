"""
Module: prep_dataset.py

This module provides functions to download and prepare the UCF50 dataset for video classification tasks.

Functions:
    - prep_dataset: Prepares the UCF50 dataset by checking for its existence in Google drive or local path
"""
import argparse
import os
import shutil
import subprocess


def prep_dataset(drive_path,local_path,url):
    """
    Prepare the UCF50 dataset by checking for its existence in Google Drive or local path,
    downloading it from the specified URL if necessary, and extracting it to the specified local path.

    Args:
        drive_path (str): Path to the datasets folder in Google Drive where rar is cached.
        local_path (str): Path to the local directory for the dataset.
        url (str): URL to download the dataset 
    """
    drive_rar = os.path.join(drive_path, 'UCF50.rar')
    local_rar = os.path.join(local_path, 'UCF50.rar')
    extract_path = os.path.join(local_path, 'datasets')
    ucf50_path = os.path.join(extract_path, 'UCF50')

    # Download the dataset from Kaggle if it doesn't exist in Google Drive or local path
    if os.path.exists(local_rar):
        print("Local rar file already exists. Skipping download.")
    elif os.path.exists(drive_rar):
        print("Copying rar file from Google Drive to local path.")
        shutil.copy(drive_rar, local_rar)
        print("Copied rar file from Google Drive to local path.")
    else:
        print("Downloading rar file from url...")
        subprocess.run(["wget","-O", local_rar+".part", url], check=True)
        subprocess.run(["mv", local_rar+".part", local_rar], check=True)
        print("Downloaded rar file from url.")
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(local_rar, drive_rar)
        print("Downloaded rar file from url and copied to Google Drive.")

    if not os.path.exists(ucf50_path):
        os.makedirs(extract_path, exist_ok=True)
        print(f"Extracting rar file to {ucf50_path}...")
        subprocess.run(["unrar", "x", "-o+","-idq", local_rar, os.path.join(extract_path,"")], check=True)
        print(f"Extracted rar file to {ucf50_path}.")
    else:
        print(f"Extract path already exists at {ucf50_path}. Skipping extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file paths for UCF50 dataset.")
    # Setup paths for dataset
    parser.add_argument('--drive_path', type=str, default='/content/drive/MyDrive/datasets/ucf50', help='Path to the datasets folder in Google Drive where zip is cached.')
    parser.add_argument('--local_path', type=str, default='/content', help='Path to the local directory for the dataset.')
    parser.add_argument('--url', type=str, default='https://www.crcv.ucf.edu/data/UCF50.rar', help='URL to download the dataset from UCF.')
    args = parser.parse_args()
    drive_path = args.drive_path
    local_path = args.local_path
    url = args.url
    prep_dataset(drive_path, local_path, url)