"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets by group, so clips from the
      same source video never appear in more than one split.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop, logs per-epoch train/val loss and accuracy and the learning rate
      to Weights & Biases, and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set, logs test loss and accuracy to Weights & Biases,
      and prints overall test accuracy, a per-class report, and per-class confusion matrices.

W&B logging can be turned off with --no-wandb, and runs can be named with --run_name.
The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN
from train import train
from evaluate import test, get_test_report, get_confusion_matrix

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        
    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -trs/--train_size: Proportion of data to use for training (default 0.7).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        --pretrained/--no-pretrained: Whether to use an ImageNet-pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
        -nw/--num_workers: Number of DataLoader worker processes (default 0; use ~CPU count on Colab).
        --wandb/--no-wandb: Whether to log the run to Weights & Biases (default True).
        --run_name: Name for the W&B run; auto-generated if not given (default None).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', default='/content/datasets/UCF50_frames_16',help='Directory containing the extracted frame folders')
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, help='Use an ImageNet-pretrained CNN backbone (disable with --no-pretrained)', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-nw','--num_workers',type=int,default=0,help="Number of workers to prepare upcoming batches")

    parser.add_argument('--wandb',action=argparse.BooleanOptionalAction,default=True,help='Use Weights and Biases to track (disable with --no-wandb)')
    parser.add_argument('--run_name', default=None,help='Name of run to log for W&B')
    args = parser.parse_args()
    return args

def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_workers = args.num_workers

    # Wandb parameters
    wb = args.wandb
    rn = args.run_name
    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Initialize the model (LRCN)
    model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
                 n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone)

    if mode == 'train':
        # Load dataset and split into train/validation/test
        vid_dataset, _ = load_dataset(frame_dir)
        tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_size, ts_size)

        # Save the splits for reproducibility and later use in evaluation
        splits = {'train': np.array(tr_split),
                  'val': np.array(val_split),
                  'test': np.array(ts_split)}
        np.save('./splits.npy', splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type,num_workers)

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        os.makedirs("./models", exist_ok=True)
        optim_model_dir = './models'
        wandb.init(
                project="Video_Action_Recognition",
                name=rn,
                config=vars(args),
                mode="online" if wb else "disabled"
                )
        # Main training procedure
        model.to(device)
        model, _, _ = train(dataloaders, model, loss_func, opt, lr_scheduler, device, optim_model_dir, n_epochs)
        wandb.finish()
    elif mode == 'eval':
        # Load saved dataset splits
        splits = np.load('./splits.npy', allow_pickle=True)
        ts_split = splits.item()['test']
        ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

        # Create PyTorch Dataset and DataLoader for the test set
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, batch_size, model_type,num_workers)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        # Load the trained model checkpoint
        model.load_state_dict(torch.load(ckpt,map_location=device))
        model.to(device)
        wandb.init(
                project="Video_Action_Recognition",
                name=rn,
                config=vars(args),
                mode="online" if wb else "disabled"
                )
        targets, outputs, accuracy, loss = test(model, dataloaders['test'], device, criterion)
        wandb.log({"test/loss":loss,"test/acc":accuracy})
        _,labels_dict = load_dataset(frame_dir)
        all_cats = sorted(labels_dict,key=labels_dict.get)
        print('The overall test accuracy is {:.4f}%.'.format(100 * accuracy))
        print('The overall test loss is {:.4f}'.format(loss))
        # Optionally, generate a detailed test report or confusion matrix:
        print(get_test_report(targets, outputs, all_cats))
        print(get_confusion_matrix(targets, outputs, labels_dict, all_cats))
        wandb.finish()
    else:
        raise ValueError('The mode argument must be either "train" or "eval".')

if __name__ == "__main__":
    args = args_parser()
    main(args)
