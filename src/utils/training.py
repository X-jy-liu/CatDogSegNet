import os
from pathlib import Path
from tqdm import tqdm
import torch
from data.preprocessing import create_point_wise_mask

#########################
# Segmentaiton Training #
#########################
def training(
    model, train_loader, val_loader,
    train_criterion, val_criterion,  # Use different loss functions 
    optimizer, num_epochs=10, device="cuda", 
    save_dir="../params", save_name=None,
    patience=5):
    """
    Train the segmentation model with early stopping.
    Args:
        model (torch.nn.Module): The segmentation model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        train_criterion (callable): Loss function for training.
        val_criterion (callable): Loss function for validation.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        device (str): Device to use ('cuda' or 'cpu').
        save_dir (str): Directory to save the model checkpoints.
        save_name (str): Name of the saved model checkpoint.
        patience (int): Number of epochs with no improvement after which training will be stopped.
    Returns:
        history (dict): Training and validation loss history.
    """
    # assgin model
    model.to(device).float()

    # print model-related info
    print("Model Structure:\n", model)
    
    # Count total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Estimate model size (in MB)
    param_size_MB = total_params * 4 / (1024 ** 2)  # float32 => 4 bytes
    print(f"Estimated Parameter Size: {param_size_MB:.2f} MB")
    
    # training utils
    history = {"train_loss": [], "val_loss": []}

    os.makedirs(save_dir, exist_ok=True)  # Create checkpoint directory
    save_path = Path(save_dir) / save_name

    # Initialize Early Stopping Variables
    best_val_loss = float("inf")  # Set to a large value
    best_model_config = None  # To store the best model parameters
    epochs_no_improve = 0  # Counter for non-improving epochs

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0

        for images, masks, _,_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit=' Batches'):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = train_criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks, _,_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit=' Batches'):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = val_criterion(outputs, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best loss
            best_epoch = epoch + 1
            epochs_no_improve = 0  # Reset counter
            best_model_config = model.state_dict()
        else:
            epochs_no_improve += 1  # Increment counter
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            # Save model
            torch.save(best_model_config, save_path)  # Save best model
            print(f"Early stopping triggered after {epoch+1} epochs!")
            print(f"Best val loss at epoch {best_epoch}: {best_val_loss:.8}")
            print(f"Best model config saved at: {save_path}")
            return history
        
    torch.save(best_model_config, save_path)  # save the model if the early stop is not triggered
    print(f"Model config saved at: {save_path} in epoch {best_epoch}")
    return history

def prompt_training(
    model, train_loader, val_loader,
    train_criterion, val_criterion,
    optimizer, num_epochs, device="cuda", 
    save_dir="../params", save_name= None,
    patience=5):
    '''
    Train the prompt-based segmentation model with early stopping.
    Args:
        model (torch.nn.Module): The segmentation model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        train_criterion (callable): Loss function for training.
        val_criterion (callable): Loss function for validation.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        device (str): Device to use ('cuda' or 'cpu').
        save_dir (str): Directory to save the model checkpoints.
        save_name (str): Name of the saved model checkpoint.
        patience (int): Number of epochs with no improvement after which training will be stopped.
    Returns:
        history (dict): Training and validation loss history.
    '''
    model.to(device).float()

    ############################
    # Print model-related info #
    ############################
    print("Model Structure:\n", model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_MB = total_params * 4 / (1024 ** 2)  # float32 => 4 bytes

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Estimated Parameter Size: {param_size_MB:.2f} MB")
    
    ##################
    # Training Setup #
    ##################
    history = {"train_loss": [], "val_loss": []}
    os.makedirs(save_dir, exist_ok=True)
    save_path = Path(save_dir) / save_name

    best_val_loss = float("inf")
    best_model_config = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit=' Batches'):
            images = batch['image'].to(device)               # (B, 3, H, W)
            gt_masks = batch['gt_mask'].to(device)           # (B, 1, H, W)
            prompt_heatmaps = batch['prompt_heatmap'].to(device) # (B, 1, H, W)
            point_classes = batch['point_class'].to(device)      # (B,)

            # Create target masks based on point class
            target_masks = create_point_wise_mask(
                point_classes, 
                gt_masks
            )

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                image=images,
                prompt_heatmap=prompt_heatmaps,
                point_class=point_classes
            )

            # Calculate loss
            loss = train_criterion(outputs, target_masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit=' Batches'):
                images = batch['image'].to(device)               # (B, 3, H, W)
                gt_masks = batch['gt_mask'].to(device)           # (B, 1, H, W)
                prompt_heatmaps = batch['prompt_heatmap'].to(device) # (B, 1, H, W)
                point_classes = batch['point_class'].to(device)      # (B,)

                    # Create target masks based on point class
                target_masks = create_point_wise_mask(
                    point_classes, 
                    gt_masks
                )

                # Forward pass
                outputs = model(
                    image=images,
                    prompt_heatmap=prompt_heatmaps,
                    point_class=point_classes
                )
                loss = val_criterion(outputs, target_masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best loss
            best_epoch = epoch + 1
            epochs_no_improve = 0  # Reset counter
            best_model_config = model.state_dict() # record the best model checkpoint so far
            # Ensure the directory exists
            tmp_save_dir = Path(save_dir) / "tmp_prompt_checkpoint"
            tmp_save_dir.mkdir(parents=True, exist_ok=True)

            # Construct save path
            tmp_save_name = f'best_prompt_checkpoint_epoch_{epoch+1}.pth'
            tmp_save_path = tmp_save_dir / tmp_save_name

            torch.save(best_model_config, tmp_save_path)
        else:
            epochs_no_improve += 1  # Increment counter
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            # Save model
            torch.save(best_model_config, save_path)  # Save best model
            print(f"Early stopping triggered after {epoch+1} epochs!")
            print(f"Best val loss at epoch {best_epoch}: {best_val_loss:.8}")
            print(f"Best model config saved at: {save_path}")
            return history
        
    torch.save(best_model_config, save_path)  # save the model if the early stop is not triggered
    print(f"Model config saved at: {save_path} (early stop not triggered)")
    return history
