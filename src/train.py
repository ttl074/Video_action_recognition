"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
logs per-epoch train/val loss and accuracy and the learning rate to Weights & Biases,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate from an optimizer.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
from dataclasses import dataclass
from tqdm import tqdm
import torch
import wandb

@dataclass
class TrainSettings:
    """
    Settings for the training loop.

    Attributes:
        device (torch.device): Device (CPU or GPU) on which to train.
        ckpt_dir (str): Directory where the best model weights are saved.
        n_epochs (int): Maximum number of training epochs.
        early_stop_patience (int): Epochs without val-loss improvement before stopping early.
    """
    device: torch.device
    ckpt_dir: str
    n_epochs: int = 30
    early_stop_patience: int = 8


@dataclass
class TrainComponents:
    """
    Objects used to optimize the model.

    Attributes:
        criterion (callable): Loss function (expected to use reduction='sum').
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): LR scheduler stepped on val loss.
    """
    criterion: object
    optimizer: torch.optim.Optimizer
    scheduler: object


def train(dataloaders, model, components, settings):
    """
    Train and validate the model over a given number of epochs.

    This function performs the training loop for a video classification model.
    It iterates through the epochs, updates model weights using backpropagation, and
    evaluates model performance on a validation set. After each epoch it logs train/val
    loss and accuracy and the learning rate to Weights & Biases (a no-op if wandb was
    initialized with mode="disabled"). The model with the best validation accuracy is
    saved to disk. When the scheduler lowers the learning rate, the best weights so far
    are reloaded before training continues. Training stops early if validation loss does
    not improve for settings.early_stop_patience consecutive epochs.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model to be trained.
        components (TrainComponents): Loss function, optimizer, and LR scheduler.
        settings (TrainSettings): Device, checkpoint directory, epochs, and early-stop patience.

    Returns:
        torch.nn.Module: The trained model loaded with the best validation weights.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float('inf')
    stop_counter = 0
    for epoch in range(settings.n_epochs):
        current_lr = get_learning_rate(components.optimizer)
        print(f'Epoch {epoch+1}/{settings.n_epochs}; Current learning rate {current_lr}')

        # Training phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(model, components.criterion,
                                                    dataloaders['train'], settings.device,
                                                    components.optimizer)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(model, components.criterion,
                                                    dataloaders['val'], settings.device)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(settings.ckpt_dir, 'best_model_wts.pt'))
            print(f'Best model weights are updated at epoch {epoch+1}!')
            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_epoch"] = epoch + 1
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss, "train/acc": train_accuracy,
                   "val/loss": val_loss, "val/acc": val_accuracy, "lr": current_lr})

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= settings.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}: "
                  f"no val-loss improvement for {stop_counter} epochs")
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
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

def get_batch_loss(criterion, output, target, optimizer=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).
    
    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights.
        If None, no backpropagation is performed.
    
    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), n_batch_correct_preds

def get_epoch_loss(model, criterion, dataloader, device, optimizer=None):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to
        update model weights during training.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        batch_loss, n_batch_correct_preds = get_batch_loss(criterion, output, y_batch, optimizer)

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy
