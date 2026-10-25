"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
logs per-epoch train/val loss and accuracy and the learning rate to Weights & Biases,
updates the best performing model based on validation accuracy, saves a resumable
checkpoint every epoch, and provides helper functions to compute the learning rate,
batch loss, and epoch loss. Training can optionally use automatic mixed precision (AMP).

Classes:
    - TrainSettings: Device, checkpoint directory, epoch limits, resume, and AMP settings.
    - TrainComponents: Loss function, optimizer, gradient scaler, and LR scheduler.
    - TrainState: Per-epoch training progress saved for resuming.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate from an optimizer.
    - save_last_checkpoint: Saves everything needed to resume training to last.pt.
    - load_last_checkpoint: Restores training from a checkpoint saved to last.pt.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and, in training, backpropagates.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
from dataclasses import dataclass, asdict
from tqdm import tqdm
import torch
import wandb


@dataclass
class TrainSettings:
    """
    Settings for the training loop.

    Attributes:
        device (torch.device): Device (CPU or GPU) on which to train.
        ckpt_dir (str): Directory where best_model_wts.pt and last.pt are saved.
        n_epochs (int): Maximum number of training epochs.
        early_stop_patience (int): Epochs without val-loss improvement before stopping early.
        resume (dict | None): Checkpoint dictionary loaded from last.pt to continue a run;
            None starts fresh.
        run_info (dict | None): Run arguments (vars(args)), saved into last.pt
            so the run can be rebuilt on resume.
        amp_dtype (torch.dtype | None): Autocast dtype (torch.bfloat16 or torch.float16);
            None disables AMP.
    """
    device: torch.device
    ckpt_dir: str
    n_epochs: int = 30
    early_stop_patience: int = 10
    resume: dict = None
    run_info: dict = None
    amp_dtype: torch.dtype = None


@dataclass
class TrainComponents:
    """
    Objects used to optimize the model.

    Attributes:
        criterion (callable): Loss function (expected to use reduction='sum').
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scaler (torch.amp.GradScaler): Loss scaler for fp16 AMP; disabled
            (pass-through) for bf16 or no AMP.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): LR scheduler stepped on
            validation loss.
    """
    criterion: object
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    scheduler: object


@dataclass
class TrainState:
    """
    Training progress that changes every epoch and is saved to last.pt for resuming.

    A fresh run starts from the defaults (TrainState()); a resumed run is rebuilt from the
    saved values (TrainState(**saved)), and the epoch loop continues at epochs_completed.

    Attributes:
        epochs_completed (int): Number of epochs fully finished, including validation and
            the scheduler step. Also the index of the next epoch to run.
        best_val_acc (float): Highest validation accuracy so far; best_model_wts.pt holds
            the weights from that epoch.
        best_val_loss (float): Lowest validation loss so far, used for early stopping.
        stop_counter (int): Consecutive epochs without a new best validation loss; training
            stops when it reaches early_stop_patience.
    """
    epochs_completed: int = 0
    best_val_acc: float = 0.0
    best_val_loss: float = float('inf')
    stop_counter: int = 0


def train(dataloaders, model, components, settings):
    """
    Train and validate the model over a given number of epochs, resuming if requested.

    This function performs the training loop for a video classification model.
    It iterates through the epochs, updates model weights using backpropagation, and
    evaluates model performance on a validation set. After each epoch it logs train/val
    loss and accuracy and the learning rate to Weights & Biases (a no-op if wandb was
    initialized with mode="disabled"). The model with the best validation accuracy is
    saved to best_model_wts.pt. When the scheduler lowers the learning rate, the best
    weights so far are reloaded before training continues. Training stops early if
    validation loss does not improve for settings.early_stop_patience consecutive epochs.

    If settings.amp_dtype is set, the forward pass and loss run under autocast in that dtype,
    and the backward pass goes through components.scaler.

    At the end of every completed epoch, a full checkpoint (model, optimizer, scheduler,
    gradient scaler, TrainState, and run info with the W&B run ID) is saved to last.pt in
    settings.ckpt_dir. If settings.resume holds a checkpoint loaded from last.pt, the
    model, optimizer, scheduler, scaler, and progress are restored from it, the best weights are
    reloaded from best_model_wts.pt, and training continues at the next epoch.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model to be trained. Must
            already be on settings.device.
        components (TrainComponents): Loss function, optimizer, gradient scaler, and LR
            scheduler.
        settings (TrainSettings): Device, checkpoint directory, epochs, early-stop
            patience, optional resume checkpoint and run info, and AMP dtype.

    Returns:
        torch.nn.Module: The trained model loaded with the best validation weights.
    """
    if settings.resume is not None:
        state = load_last_checkpoint(settings.resume, model, components)
        best_model_wts = torch.load(os.path.join(settings.ckpt_dir, 'best_model_wts.pt'),
                                    map_location=settings.device)
        print(f"Resuming at epoch {state.epochs_completed + 1}")
    else:
        state = TrainState()
        best_model_wts = copy.deepcopy(model.state_dict())
    run_info = {**(settings.run_info or {}), "wandb_id": wandb.run.id}
    for epoch in range(state.epochs_completed, settings.n_epochs):
        current_lr = get_learning_rate(components.optimizer)
        print(f'Epoch {epoch+1}/{settings.n_epochs}; Current learning rate {current_lr}')

        # Training phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(model, dataloaders['train'], components,
                                                    settings, True)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(model, dataloaders['val'], components,
                                                    settings, False)
        if val_accuracy > state.best_val_acc:
            state.best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(settings.ckpt_dir, 'best_model_wts.pt'))
            print(f'Best model weights are updated at epoch {epoch+1}!')
            wandb.run.summary["best_val_acc"] = state.best_val_acc
            wandb.run.summary["best_epoch"] = epoch + 1
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss, "train/acc": train_accuracy,
                   "val/loss": val_loss, "val/acc": val_accuracy, "lr": current_lr})

        # Early stopping on validation loss
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            state.stop_counter = 0
        else:
            state.stop_counter += 1
        if state.stop_counter >= settings.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}: "
                  f"no val-loss improvement for {state.stop_counter} epochs")
            wandb.run.summary["stopped_epoch"] = epoch + 1
            break

        # Update learning rate based on validation loss
        components.scheduler.step(val_loss)
        if current_lr != get_learning_rate(components.optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, "
              f"accuracy: {100*val_accuracy:.2f}")
        print("-" * 60)
        print()
        state.epochs_completed = epoch + 1
        save_last_checkpoint(os.path.join(settings.ckpt_dir, 'last.pt'),
                             model, components, state, run_info)

    # Load the best model weights before returning the model
    model.load_state_dict(best_model_wts)
    return model


def get_learning_rate(optimizer):
    """
    Retrieve the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rate.

    Returns:
        float: The current learning rate.
    """
    return optimizer.param_groups[0]['lr']


def save_last_checkpoint(path, model, components, state, run_info):
    """
    Save everything needed to resume training to path (typically ckpt_dir/last.pt).

    The checkpoint is written to path + '.tmp' first and then moved onto path with
    os.replace, which replaces the file in a single step. If the session disconnects
    mid-save, the previous last.pt stays intact instead of being left half-written.

    Args:
        path (str): Destination file for the checkpoint.
        model (torch.nn.Module): The model whose weights are saved.
        components (TrainComponents): Supplies the optimizer, scheduler, and gradient
            scaler, whose state_dicts are saved so learning-rate adjustments, optimizer
            statistics, and the fp16 loss scale carry over.
        state (TrainState): Epochs completed, best validation metrics, and early-stopping
            counter.
        run_info (dict): Run arguments (vars(args)) and W&B run ID, used to rebuild the
            run and continue its W&B logging on resume.

    Returns:
        None
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": components.optimizer.state_dict(),
        "scheduler": components.scheduler.state_dict(),
        "state": asdict(state),
        "run_info": run_info,
        "scaler": components.scaler.state_dict(),
    }
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_last_checkpoint(ckpt, model, components):
    """
    Restore the model, optimizer, scheduler, scaler, and training progress from a checkpoint.

    Restores the model weights, AdamW's per-parameter statistics (so updates continue
    smoothly instead of restarting cold), the ReduceLROnPlateau state (its best loss and
    bad-epoch count, so pending learning-rate drops happen on schedule), the GradScaler's
    loss scale (only if saved: checkpoints from before AMP support, and runs with the
    scaler disabled, have none), and the TrainState counters. The current learning rate
    is restored as part of the optimizer state.

    The model must already be on its training device: optimizer.load_state_dict moves the
    restored statistics onto the device of the model's parameters at the time it is called.

    Args:
        ckpt (dict): An already-loaded checkpoint dictionary written by save_last_checkpoint.
        model (torch.nn.Module): A model with the same architecture as the saved one.
        components (TrainComponents): An optimizer built over the same trainable parameters
            in the same order as the saved one, its scheduler, and a gradient scaler.

    Returns:
        TrainState: Training progress to continue from (the loop resumes at
            epochs_completed).
    """
    model.load_state_dict(ckpt["model"])
    components.optimizer.load_state_dict(ckpt["optimizer"])
    components.scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler"):
        components.scaler.load_state_dict(ckpt["scaler"])
    return TrainState(**ckpt["state"])


def batch_correct_preds(output, target):
    """
    Compute the number of correct predictions for a mini-batch.

    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).

    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds


def get_batch_loss(components, output, target, amp_dtype, training):
    """
    Compute the loss for a mini-batch and, in training, backpropagate and update the weights.

    The loss is computed under autocast when AMP is on, so cross-entropy runs in fp32 even
    when the model outputs are fp16/bf16. The backward pass runs outside autocast, and goes
    through components.scaler (a pass-through unless training with fp16).

    Args:
        components (TrainComponents): Holds the criterion, optimizer, and scaler.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        amp_dtype (torch.dtype | None): Autocast dtype; None disables AMP.
        training (bool): If True, zero the gradients, backpropagate, and step the optimizer;
            if False (validation), only compute the loss and correct predictions.

    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        loss = components.criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if training:
        components.optimizer.zero_grad()
        components.scaler.scale(loss).backward()
        components.scaler.step(components.optimizer)
        components.scaler.update()
    return loss.item(), n_batch_correct_preds


def get_epoch_loss(model, dataloader, components, settings, training):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch. The forward pass runs under autocast when
    settings.amp_dtype is set. The caller sets model.train()/model.eval() and wraps
    validation in torch.no_grad().

    Args:
        model (torch.nn.Module): The video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        components (TrainComponents): Holds the criterion, optimizer, and scaler.
        settings (TrainSettings): Supplies the device and AMP dtype.
        training (bool): If True, update the model weights on every mini-batch; if False
            (validation), only compute the loss and accuracy.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(settings.device), y_batch.to(settings.device)
        with torch.autocast("cuda", dtype=settings.amp_dtype,
                            enabled=settings.amp_dtype is not None):
            output = model(x_batch)
        batch_loss, n_batch_correct_preds = get_batch_loss(components, output, y_batch,
                                                           settings.amp_dtype, training)

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy
