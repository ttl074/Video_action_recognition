"""
Module: prep_dataset.py

This module provides functions to download and prepare the HMDB51 dataset for video classification tasks.

Functions:
    - setup_kaggle_api: Sets up Kaggle API credentials for downloading datasets.
    - prep_dataset: Prepares the HMDB51 dataset by checking for its existence in 
    Google Drive or local path, downloading it from Kaggle if necessary, and extracting
    it to the specified local extract path.
"""
import argparse
import os
import shutil
import subprocess


def setup_kaggle_api(kaggle_token):
    """
    Set up Kaggle API credentials by copying the access token to the appropriate location
    and setting the correct permissions.

    Args:
        kaggle_token (str): Path to the Kaggle API token file.
    """
    config_dir = os.path.expanduser("~/.kaggle")
    access_token_path = os.path.join(config_dir, "access_token")
    os.makedirs(config_dir, exist_ok=True)
    shutil.copy(kaggle_token, access_token_path)
    os.chmod(access_token_path, 0o600)  # Set permissions to read/write for the user only


def prep_dataset(drive_path,local_path,kaggle_token):
    """
    Prepare the HMDB51 dataset by checking for its existence in Google Drive or local path,
    downloading it from Kaggle if necessary, and extracting it to the specified local path.

    Args:
        drive_path (str): Path to the datasets folder in Google Drive where zip is cached.
        local_path (str): Path to the local directory for the dataset.
        kaggle_token (str): Path to the Kaggle API token file.
    """
    drive_zip = os.path.join(drive_path, 'hmdb51.zip')
    local_zip = os.path.join(local_path, 'hmdb51.zip')
    extract_path = os.path.join(local_path, 'datasets')
    hmdb51_path = os.path.join(extract_path, 'HMDB51')

    # Download the dataset from Kaggle if it doesn't exist in Google Drive or local path
    if os.path.exists(local_zip):
        print("Local zip file already exists. Skipping download.")
    elif os.path.exists(drive_zip):
        shutil.copy(drive_zip, local_zip)
        print("Copied zip file from Google Drive to local path.")
    else:
        # Setup Kaggle API credentials
        setup_kaggle_api(kaggle_token)
        subprocess.run(["kaggle", "datasets", "download", "-d", "easonlll/hmdb51", "-p", local_path],check=True)
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(local_zip, drive_zip)
        print("Downloaded zip file from Kaggle and copied to Google Drive.")

    if not os.path.exists(hmdb51_path):
        subprocess.run(["unzip", "-q", local_zip, "-d", extract_path], check=True)
        print(f"Extracted zip file to {hmdb51_path}.")
    else:
        print(f"Extract path already exists at {hmdb51_path}. Skipping extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file paths for HMDB51 dataset.")
    # Setup paths for dataset
    parser.add_argument('--drive_path', type=str, default='/content/drive/MyDrive/datasets/hmdb51', help='Path to the datasets folder in Google Drive where zip is cached.')
    parser.add_argument('--local_path', type=str, default='/content', help='Path to the local directory for the dataset.')
    parser.add_argument('--kaggle_token', type=str, default='/content/drive/MyDrive/credentials/Kaggle_API.txt', help='Path to the Kaggle API token file.')
    args = parser.parse_args()
    drive_path = args.drive_path
    local_path = args.local_path
    kaggle_token = args.kaggle_token
    prep_dataset(drive_path, local_path,kaggle_token)