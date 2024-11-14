# checkpoint.py

import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, file_path='checkpoint.pth', scheduler=None):
    """
    Saves model and optimizer state to a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        epoch (int): Current epoch number
        loss (float): Current loss value
        file_path (str): File path to save the checkpoint
        scheduler: Optional learning rate scheduler
    """
    if scheduler is None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }
    else:
         checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} to {file_path}")

def load_checkpoint(model, optimizer, file_path='checkpoint.pth', scheduler=None):
    """
    Loads model and optimizer state from a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model to load state into
        optimizer (torch.optim.Optimizer): The optimizer to load state into
        file_path (str): File path of the checkpoint to load
        scheduler: Optional learning rate scheduler
    
    Returns:
        int: The epoch at which the checkpoint was saved
        float: The loss value at the saved checkpoint
    """
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded from {file_path} at epoch {epoch}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {file_path}")
        return None, None
