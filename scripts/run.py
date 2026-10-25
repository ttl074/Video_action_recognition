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
    - Initializes the model (an LRCN or an R3D-18 3D CNN, chosen with --model_type) and
      defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop, logs per-epoch train/val loss and accuracy and the learning rate
      to Weights & Biases, and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits and the trained model checkpoint.
    - With one --test_offsets value (default 0.5), runs single-clip testing: one clip per
      video, sampled at that position within each temporal segment.
    - With more than one value, runs multi-clip testing: one clip per offset, with softmax
      probabilities averaged across clips before taking the prediction.
    - Logs test loss and accuracy to Weights & Biases, and prints overall test accuracy, a
      per-class report, and per-class confusion matrices.

W&B logging can be turned off with --no-wandb, and runs can be named with --run_name.
An interrupted training run can be continued from its last.pt checkpoint with --resume.
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
from models import LRCN, LRCNConfig, build_r3d18
from train import train, TrainComponents, TrainSettings
from evaluate import test, get_test_report, get_confusion_matrix, predict_probs

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
        -mt/--model_type: Model type: 'lrcn' (ResNet + LSTM) or '3dcnn' (R3D-18)
            (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction; LRCN only
            (default 'resnet34').
        --pretrained/--no-pretrained: Whether to start from ImageNet-pretrained weights
            (default True). LRCN loads them into its 2D backbone; R3D-18 uses ResNet18's,
            inflated along time.
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer; LRCN only
            (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers; LRCN only (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Maximum number of training epochs (default 30).
        -nw/--num_workers: Number of DataLoader worker processes (default 0).
        --early_stop_patience: Epochs without val-loss improvement before stopping (default 10).
        --weight_decay: Weight decay for AdamW optimizer (default 0.0).
        --freeze_layers: Layers of the ResNet backbone to freeze; LRCN only
            (default none frozen).
        --wandb/--no-wandb: Whether to log the run to Weights & Biases (default True).
        --run_name: Name for the W&B run; auto-generated if not given (default None).
        --resume: Path to a last.pt checkpoint to resume training from (default None).
        --test_offsets: Fractional positions in [0, 1) within each temporal segment for
            test clips; more than one averages predictions over clips (multi-clip testing)
            (default 0.5).
        --amp/--no-amp: Whether to train with automatic mixed precision (default False).
            Uses bf16 on GPUs with native support (e.g. A100) and fp16 with a GradScaler
            otherwise (e.g. T4).
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
    parser.add_argument('-mt', '--model_type', choices=['lrcn', '3dcnn'], default='lrcn',
                        help="'lrcn' (ResNet + LSTM) or '3dcnn' (R3D-18)")
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='2D CNN backbone')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Start from ImageNet weights (LRCN: 2D backbone; '
                        '3dcnn: inflated ResNet18) (disable with --no-pretrained)')
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
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Number of epochs with no val-loss improvement before early stop')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for AdamW optimizer (0 disables)')
    parser.add_argument('--freeze_layers', nargs='*',
                        choices=['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'],
                        help='Layers of CNN backbone to freeze')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Log the run to Weights & Biases (disable with --no-wandb)')
    parser.add_argument('--run_name', default=None, help='Name of the run to log in W&B')
    parser.add_argument('--resume', default=None, help='Path to last.pt for resuming training')
    parser.add_argument('--test_offsets', type=float, nargs='+', default=[0.5],
                        help='Fractional positions in [0, 1) within each segment for test '
                        'clips; more than one enables multi-clip testing')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help='Use automatic mixed precision for training (bf16 on GPUs that '
                        'support it natively, fp16 with gradient scaling otherwise)')
    return parser.parse_args()


def restore_args(args, ckpt):
    """
    Rebuild the run's arguments from a checkpoint for resuming training.

    Precedence, lowest to highest: the current command's arguments (so flags added
    after the checkpoint was saved still get their defaults), then the saved run's
    arguments from ckpt["run_info"] (so model, data, and training settings match the
    original run), then the session-specific values resume, num_workers, and wandb from
    the current command (which depend on the machine and session, not the experiment).
    The bookkeeping key wandb_id is dropped. ckpt is not modified.

    Args:
        args (argparse.Namespace): Arguments parsed from the current command.
        ckpt (dict): Checkpoint loaded from last.pt by save_last_checkpoint.

    Returns:
        argparse.Namespace: Arguments for the resumed run.
    """
    args_dict = dict(vars(args))
    args_dict.update(ckpt["run_info"])
    args_dict.pop("wandb_id", None)
    for attr in ("resume", "num_workers", "wandb"):
        args_dict[attr] = getattr(args, attr)
    return argparse.Namespace(**args_dict)


def build_model(args):
    """
    Build the model selected by --model_type from the command-line arguments.

    For 'lrcn', builds an LRCN from the RNN, backbone, dropout, pretrained, and
    freeze_layers arguments. For '3dcnn', builds an R3D-18 with build_r3d18, which uses only
    n_classes, dropout, and pretrained (inflated ImageNet ResNet18 weights); the backbone,
    RNN, and freeze_layers arguments are ignored.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.nn.Module: An LRCN or an R3D-18 (VideoResNet) model, untrained or initialized
            from ImageNet weights.
    """
    if args.model_type == 'lrcn':
        config = LRCNConfig(hidden_size=args.rnn_hidden_size, n_layers=args.rnn_n_layers,
                            dropout_rate=args.dropout, n_classes=args.n_classes,
                            pretrained=args.pretrained, cnn_model=args.cnn_backbone,
                            freeze_layers=tuple(args.freeze_layers or ()))
        return LRCN(config)
    return build_r3d18(args.n_classes, args.dropout, args.pretrained)


def setup_amp(args, device):
    """
    Choose the mixed-precision settings for training from --amp and the GPU's capabilities.

    AMP is off when --amp is not given or when training on CPU. On GPUs with native bf16
    support (compute capability 8.0 or higher, e.g. A100), uses bf16, which has the same
    exponent range as fp32, so gradients do not underflow and no loss scaling is needed.
    On older GPUs (e.g. T4), uses fp16 with an enabled GradScaler to keep small gradients
    from underflowing to zero.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (uses --amp).
        device (torch.device): Device on which to train.

    Returns:
        tuple: (amp_dtype, scaler):
            - amp_dtype (torch.dtype or None): torch.bfloat16 or torch.float16 for autocast,
              or None when AMP is off.
            - scaler (torch.amp.GradScaler): Enabled only for fp16; when disabled, its
              scale/step/update calls pass through unchanged, so the training loop needs
              no AMP-specific branches.
    """
    if not args.amp or device.type != "cuda":
        return None, torch.amp.GradScaler("cuda", enabled=False)
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    return torch.float16, torch.amp.GradScaler("cuda")


def init_wandb(args, run_id=None):
    """
    Start a Weights & Biases run, or a disabled no-op run when --no-wandb is given.
    If run_id given, run is resumed and fails if run_id does not exist.

    Args:
        args (argparse.Namespace): Parsed command-line arguments; all are recorded as the
            run config.
        run_id (str, optional): ID of an existing W&B run to continue when resuming;
            None starts a new run.
    """
    wandb.init(project=WANDB_PROJECT, name=args.run_name, config=vars(args),
               mode="online" if args.wandb else "disabled", id=run_id,
               resume="must" if run_id else None)


def prep_dataloaders(args, transforms):
    """
    Load the frame dataset, split it by group, save the splits, and build DataLoaders.

    Splits the videos into train/validation/test sets by group, saves all three splits to
    SPLITS_PATH (later read by run_eval), and builds the training and validation
    DataLoaders. The training dataset samples one frame at a random position within each
    temporal segment; the validation dataset uses segment centers.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        transforms (tuple): (train_transforms, val_test_transforms) from
            compose_data_transforms.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
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
    dataloaders = train_val_dloaders(tr_dataset, val_dataset,
                                     args.batch_size, args.model_type, args.num_workers)
    return dataloaders


def run_train(args, model, device, transforms, ckpt=None):
    """
    Split the dataset by group, train the model, and save the best weights.

    Builds the AdamW optimizer over the trainable (non-frozen) parameters only, a
    ReduceLROnPlateau scheduler on validation loss (factor 0.5, patience 4), and summed
    cross-entropy loss. When resuming, continues the original W&B run using the run ID
    saved in the checkpoint.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The model to train.
        device (torch.device): Device (CPU or GPU) on which to train.
        transforms (tuple): (train_transforms, val_test_transforms) from
            compose_data_transforms.
        ckpt (dict, optional): Checkpoint loaded from last.pt to resume from;
            None starts a fresh run.

    Notes:
        Best weights (best_model_wts.pt) and the resume checkpoint (last.pt) are saved to
        models/<run_name>/, or models/ if no run name is given.
    """
    dataloaders = prep_dataloaders(args, transforms)
    # Define the loss function, optimizer, and learning rate scheduler
    non_frozen = [param for param in model.parameters() if param.requires_grad]
    opt = optim.AdamW(non_frozen, lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_dtype, scaler = setup_amp(args, device)
    components = TrainComponents(criterion=nn.CrossEntropyLoss(reduction='sum'),
                                 optimizer=opt, scaler=scaler,
                                 scheduler=ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                             patience=4))
    if args.run_name:
        ckpt_dir = os.path.join(CKPT_DIR, args.run_name)
    else:
        ckpt_dir = CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    settings = TrainSettings(device=device, ckpt_dir=ckpt_dir, n_epochs=args.n_epochs,
                             early_stop_patience=args.early_stop_patience,
                             resume=ckpt, run_info=vars(args), amp_dtype=amp_dtype)
    # Main training procedure
    run_id = ckpt["run_info"]["wandb_id"] if ckpt is not None else None
    init_wandb(args, run_id)
    model.to(device)
    train(dataloaders, model, components, settings)
    wandb.finish()


def multiclip_eval(args, model, device, ts_split, val_ts_transforms):
    """
    Evaluate the model on the test split with multi-clip testing.

    For each offset in args.test_offsets, samples one clip per video at that position
    within each temporal segment, runs the model, and prints that clip's own accuracy.
    The softmax probabilities are then averaged across clips, and the final prediction
    for each video is the class with the highest averaged probability.

    Args:
        args (argparse.Namespace): Parsed arguments; uses test_offsets, fr_per_vid,
            batch_size, model_type, and num_workers.
        model (torch.nn.Module): Model with the checkpoint already loaded.
        device (torch.device): Device to run inference on.
        ts_split (list): Test samples as (frame_dir, label) tuples.
        val_ts_transforms (callable): Validation/test transforms from compose_data_transforms.

    Returns:
        tuple: (labels, preds, avg_acc, avg_loss), in the same order as evaluate.test:
            - labels (list[int]): True class index for each video.
            - preds (list[int]): Predicted class index from the averaged probabilities.
            - avg_acc (float): Fraction of videos predicted correctly.
            - avg_loss (float): Mean negative log-likelihood of the true class under
              the averaged probabilities.

    Raises:
        ValueError: If any offset is outside [0.0, 1.0).

    Notes:
        - args.test_offsets must contain at least one offset (guaranteed by nargs='+').
        - Every pass must return videos in the same order so probabilities line up across
          clips; test_dloaders uses shuffle=False.
        - The 0.5 pass reproduces the single-clip (segment-center) evaluation exactly.
    """
    probs = []
    for offset in args.test_offsets:
        if offset >= 1.0 or offset < 0.0:
            raise ValueError(f"Offset must be in [0.0, 1.0), got {offset}")
        vd = VideoDataset(ts_split, args.fr_per_vid, False, val_ts_transforms, offset=offset)
        dataloader = test_dloaders(vd, args.batch_size, args.model_type,
                                   args.num_workers)['test']
        labels, prob = predict_probs(model, dataloader, device)
        probs.append(prob)
        print(f"offset {offset}: {100 * (prob.argmax(axis=1) == np.array(labels)).mean():.2f}%")
    stacked_avg = np.stack(probs).mean(axis=0)
    preds = stacked_avg.argmax(axis=1).tolist()
    avg_acc = (np.array(preds) == np.array(labels)).mean()
    avg_loss = -np.log(stacked_avg[np.arange(len(labels)), labels] + 1e-12).mean()
    return labels, preds, avg_acc, avg_loss


def run_eval(args, model, device, val_ts_transforms):
    """
    Evaluate a trained checkpoint on the saved test split and log test metrics.

    With one value in --test_offsets, runs standard single-clip testing: one clip per
    video, sampled at that position within each temporal segment (default 0.5, the
    segment center). With more than one, runs multi-clip testing via multiclip_eval:
    one clip per offset, with softmax probabilities averaged across clips before taking
    the prediction.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (uses --ckpt,
            --test_offsets, --fr_per_vid, --batch_size, --model_type, --num_workers,
            and --frame_dir).
        model (torch.nn.Module): A model with the same architecture as the checkpoint.
        device (torch.device): Device (CPU or GPU) on which to evaluate.
        val_ts_transforms (callable): Validation/test transforms from compose_data_transforms.

    Notes:
        - Single-clip results are logged to W&B as test/acc and test/loss; multi-clip
          results as test/acc_multiclip and test/loss_multiclip. test/n_clips records
          the number of clips either way.
        - Prints overall accuracy and loss, the per-class classification report, and the
          per-class confusion matrices.
    """
    # Load saved dataset splits
    splits = np.load(SPLITS_PATH, allow_pickle=True)
    ts_split = [(sample[0], int(sample[1])) for sample in splits.item()['test']]

    # Load the trained model checkpoint
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    init_wandb(args)
    if len(args.test_offsets) > 1:
        suffix = "_multiclip"
        targets, outputs, accuracy, loss = multiclip_eval(
            args, model, device, ts_split, val_ts_transforms)
    else:
        suffix = ""
        # Create PyTorch Dataset and DataLoader for the test set
        dataloaders = test_dloaders(
            VideoDataset(ts_split, args.fr_per_vid, False, val_ts_transforms,
                         offset=args.test_offsets[0]),
            args.batch_size, args.model_type, args.num_workers)
        targets, outputs, accuracy, loss = test(model, dataloaders['test'], device,
                                                nn.CrossEntropyLoss(reduction='sum'))
    wandb.log({f"test/loss{suffix}": loss, f"test/acc{suffix}": accuracy,
               "test/n_clips": len(args.test_offsets)})

    # Class names in label order (load_dataset numbers classes by sorted folder name)
    _, labels_dict = load_dataset(args.frame_dir)
    all_cats = sorted(labels_dict, key=labels_dict.get)
    print(f'The overall {len(args.test_offsets)}-clip test accuracy is {100 * accuracy:.4f}%.')
    print(f'The overall {len(args.test_offsets)}-clip test loss is {loss:.4f}')
    print(get_test_report(targets, outputs, all_cats))
    print(get_confusion_matrix(targets, outputs, labels_dict, all_cats))
    wandb.finish()


def main(args):
    """
    Build the transforms and model, then run training or evaluation based on args.mode.

    If --resume is given, loads that last.pt checkpoint first and restores the original
    run's arguments with restore_args, so the transforms, model, and data match the
    interrupted run; the checkpoint is then passed to run_train to continue training
    from the next epoch. A resumed run always trains, since the restored mode is 'train'.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        args = restore_args(args, ckpt)
    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(args.model_type)
    transforms = compose_data_transforms(h, w, mean, std)

    model = build_model(args)

    if args.mode == 'train':
        run_train(args, model, device, transforms, ckpt)
    else:
        run_eval(args, model, device, transforms[1])


if __name__ == "__main__":
    main(args_parser())
