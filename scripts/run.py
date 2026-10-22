"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model
parameters, and training hyperparameters. Depending on the selected mode
('train' or 'eval'), it performs the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets by group, so clips from the
      same source video never appear in more than one split.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop, logs per-epoch train/val loss and accuracy and the learning rate
      to Weights & Biases, and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set, logs test loss and accuracy to Weights & Biases,
      and prints overall test accuracy, a per-class report, and per-class confusion matrices.

W&B logging can be turned off with --no-wandb, and runs can be named with --run_name.
"""

import os
import argparse
import numpy as np
import wandb

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN, LRCNConfig
from train import train, TrainComponents, TrainSettings
from evaluate import test, get_test_report, get_confusion_matrix

WANDB_PROJECT = "Video_Action_Recognition"
CKPT_DIR = "./models"
SPLITS_PATH = "./splits.npy"


def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments include:
        -fd/--frame_dir: Directory containing the extracted frame folders.
        -trs/--train_size: Proportion of groups to use for training (default 0.7).
        -tss/--test_size: Proportion of groups to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        --pretrained/--no-pretrained: Whether to use an ImageNet-pretrained CNN backbone
            (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Maximum number of training epochs (default 30).
        -nw/--num_workers: Number of DataLoader worker processes (default 0).
        --early_stop_patience: Epochs without val-loss improvement before stopping (default 8).
        --wandb/--no-wandb: Whether to log the run to Weights & Biases (default True).
        --run_name: Name for the W&B run; auto-generated if not given (default None).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', default='/content/datasets/UCF50_frames',
                        help='Directory containing the extracted frame folders')
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16,
                        help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int,
                        help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='2D CNN backbone')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Use an ImageNet-pretrained CNN backbone '
                        '(disable with --no-pretrained)')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100,
                        help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1,
                        help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, choices=['train', 'eval'], required=True,
                        help="Either 'train' or 'eval'")
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5,
                        help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30,
                        help='Maximum number of training epochs')
    parser.add_argument('-nw', '--num_workers', type=int, default=0,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--early_stop_patience', type=int, default=8,
                        help='Number of epochs with no val-loss improvement before early stop')

    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Log the run to Weights & Biases (disable with --no-wandb)')
    parser.add_argument('--run_name', default=None, help='Name of the run to log in W&B')
    return parser.parse_args()


def build_model(args):
    """
    Build an LRCN model from the command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        LRCN: The constructed (untrained or pretrained-backbone) model.
    """
    config = LRCNConfig(hidden_size=args.rnn_hidden_size, n_layers=args.rnn_n_layers,
                        dropout_rate=args.dropout, n_classes=args.n_classes,
                        pretrained=args.pretrained, cnn_model=args.cnn_backbone)
    return LRCN(config)


def init_wandb(args):
    """
    Start a Weights & Biases run, or a disabled no-op run when --no-wandb is given.

    Args:
        args (argparse.Namespace): Parsed command-line arguments; all are recorded as the
            run config.
    """
    wandb.init(project=WANDB_PROJECT, name=args.run_name, config=vars(args),
               mode="online" if args.wandb else "disabled")


def run_train(args, model, device, transforms):
    """
    Split the dataset by group, train the model, and save the best weights.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The model to train.
        device (torch.device): Device (CPU or GPU) on which to train.
        transforms (tuple): (train_transforms, val_test_transforms) from
            compose_data_transforms.
    """
    tr_transforms, val_ts_transforms = transforms

    # Load dataset and split into train/validation/test by group
    vid_dataset, _ = load_dataset(args.frame_dir)
    tr_split, val_split, ts_split = dataset_split(vid_dataset, args.train_size, args.test_size)

    # Save the splits for reproducibility and later use in evaluation
    np.save(SPLITS_PATH, {'train': np.array(tr_split),
                          'val': np.array(val_split),
                          'test': np.array(ts_split)})
    # Create PyTorch Datasets and DataLoaders for train and validation
    tr_dataset = VideoDataset(tr_split, args.fr_per_vid, True, tr_transforms)
    val_dataset = VideoDataset(val_split, args.fr_per_vid, False, val_ts_transforms)
    dataloaders = train_val_dloaders(tr_dataset,val_dataset,
                                     args.batch_size, args.model_type, args.num_workers)

    # Define the loss function, optimizer, and learning rate scheduler
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    components = TrainComponents(criterion=nn.CrossEntropyLoss(reduction='sum'),
                                 optimizer=opt,
                                 scheduler=ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                             patience=5))
    os.makedirs(CKPT_DIR, exist_ok=True)
    settings = TrainSettings(device=device, ckpt_dir=CKPT_DIR, n_epochs=args.n_epochs,
                             early_stop_patience=args.early_stop_patience)

    # Main training procedure
    init_wandb(args)
    model.to(device)
    train(dataloaders, model, components, settings)
    wandb.finish()


def run_eval(args, model, device, val_ts_transforms):
    """
    Evaluate a trained checkpoint on the saved test split and log test metrics.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (uses --ckpt).
        model (torch.nn.Module): A model with the same architecture as the checkpoint.
        device (torch.device): Device (CPU or GPU) on which to evaluate.
        val_ts_transforms (callable): Validation/test transforms from compose_data_transforms.
    """
    # Load saved dataset splits
    splits = np.load(SPLITS_PATH, allow_pickle=True)
    ts_split = [(sample[0], int(sample[1])) for sample in splits.item()['test']]

    # Create PyTorch Dataset and DataLoader for the test set
    dataloaders = test_dloaders(VideoDataset(ts_split, args.fr_per_vid, False, val_ts_transforms),
                                args.batch_size, args.model_type, args.num_workers)

    # Load the trained model checkpoint
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    init_wandb(args)
    targets, outputs, accuracy, loss = test(model, dataloaders['test'], device,
                                            nn.CrossEntropyLoss(reduction='sum'))
    wandb.log({"test/loss": loss, "test/acc": accuracy})

    # Class names in label order (load_dataset numbers classes by sorted folder name)
    _, labels_dict = load_dataset(args.frame_dir)
    all_cats = sorted(labels_dict, key=labels_dict.get)
    print(f'The overall test accuracy is {100 * accuracy:.4f}%.')
    print(f'The overall test loss is {loss:.4f}')
    print(get_test_report(targets, outputs, all_cats))
    print(get_confusion_matrix(targets, outputs, labels_dict, all_cats))
    wandb.finish()


def main(args):
    """
    Build the transforms and model, then run training or evaluation based on args.mode.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(args.model_type)
    transforms = compose_data_transforms(h, w, mean, std)

    model = build_model(args)

    if args.mode == 'train':
        run_train(args, model, device, transforms)
    else:
        run_eval(args, model, device, transforms[1])


if __name__ == "__main__":
    main(args_parser())
