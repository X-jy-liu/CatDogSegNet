# run_inference.py

import torch
from data.preprocessing import (class2color, 
                                resize_with_padding, 
                                clip_transform, 
                                standard_transform,
                                create_point_wise_mask)
import os
from utils.restore_mask_size import restore_original_mask
from data.PetDataset import PetDataset, PetDatasetWithPrompt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

def custom_collate_fn(batch):
    images, masks, initial_img_sizes, filenames = zip(*batch)  # Unpack batch

    # Convert images and masks to tensors (PyTorch default behavior)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    return images, masks, list(initial_img_sizes), list(filenames)  # Keep `initial_img_sizes` as list of tuples

def inference(image_dir: str, mask_dir: str, save_dir: str, input_image_size: int,
                  mode: int, model, device: str) -> np.ndarray:
    """
    Perform inference on input images and save the predicted segmentation masks.

    Args:
        image_dir (str): Directory containing input images.
        mask_dir (str): Directory containing ground truth masks (for size reference).
        save_dir (str): Directory to save the output predicted masks.
        input_image_size (int): Target size for resizing (used in CLIP mode).
        mode (int): Model mode selector:
                    0 - U-Net,
                    1 - Autoencoder,
                    2 - CLIP.
        model (torch.nn.Module): Pre-loaded segmentation model ready for inference.
        device (str): Computation device ("cuda" or "cpu").

    Returns:
        None. (Predicted color masks are saved to 'save_dir')
    """
    
    # Initialize paths of images (list of str)
    image_paths = sorted(Path(image_dir).glob("*.*"))
    mask_paths = sorted(Path(mask_dir).glob("*.*"))
    
    # assign the correct torch dataset format
    if mode == 0 or mode == 1:
        test_dataset = PetDataset(
        img_paths = image_paths,
        msk_paths = mask_paths,
        resize_fn= resize_with_padding,
        resize_target_size= 512,
        transform = standard_transform)
    elif mode == 2:
        test_dataset = PetDataset(
            img_paths = image_paths,
            msk_paths = mask_paths,
            resize_fn = resize_with_padding,
            resize_target_size = input_image_size,
            transform = clip_transform
        )

    else:
        raise ValueError("Invalid mode. Use 0 for U-Net, 1 for autoencoder-based segmentation or 2 for CLIP-based segmentation.")

    test_loader = DataLoader(test_dataset,batch_size=4, num_workers=4, collate_fn=custom_collate_fn)


    with torch.no_grad():
        total_batches = len(test_loader)  # Get total number of batches
        for images, _, initial_img_sizes, filenames in \
        tqdm(test_loader, desc=f"Processing {len(test_loader.dataset)} images in {total_batches} batches"):
            images = images.to(device)
            outputs = model(images)

            # Convert logits to class labels (B, C, H, W) → (B, H, W)
            pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
            resized_pred_masks = [restore_original_mask(pred_masks[i], initial_img_sizes[i]) for i in range(len(images))]

            # Save predicted masks
            for i, filename in enumerate(filenames):
                pred_mask_img = class2color(resized_pred_masks[i])  # Convert class labels to color mask
                # Ensure data type is uint8 to avoid artifacts
                pred_mask_img = np.array(pred_mask_img, dtype=np.uint8)  
                save_img = Image.fromarray(pred_mask_img, mode="RGB") # Convert the ndarray into RGB Image 
                save_path = os.path.join(save_dir, Path(filename).stem+'.png')  # Keep original filename
                save_img.save(save_path,format='PNG')
    
    # print the dst directory
    print(f"Predicted masks saved to {save_dir}")

def promptInference(image_dir: str, mask_dir:str, point_dir:str, 
                    gt_save_dir:str, pred_save_dir:str,
                    threshold:float, input_image_size: int, 
                    model: callable, device: str) -> np.ndarray:
    '''
    Run the inference on test images with input of prompt points
    
    Args:
        image_dir (str): Directory containing input images
        mask_dir (str): Directory containing ground truth mask images
        point_dir (str): Directory containing point prompts
        gt_save_dir (str): Directory where ground truth masks will be saved
        pred_save_dir (str): Directory where predicted masks will be saved
        threshold (float): Threshold value for binary segmentation (sigmoid output > threshold)
        input_image_size (int): Target size for resizing images before inference
        model (callable): The segmentation model to use for inference
        device (str): Device to run inference on (e.g., 'cuda', 'cpu')
    
    Returns:
        np.ndarray: Processed data array containing segmentation results
    
    Note:
        - The function processes batches of images with corresponding masks and point prompts
        - Predicted masks are saved as PNG files in pred_save_dir
        - Ground truth masks are saved as PNG files in gt_save_dir
        - File names are preserved from the original image files
    '''
    image_paths = sorted(Path(image_dir).glob("*.*"))
    mask_paths = sorted(Path(mask_dir).glob("*.*"))
    point_paths = sorted(Path(point_dir).glob("*.*"))

    test_dataset = PetDatasetWithPrompt(
        img_paths=image_paths,
        msk_paths=mask_paths,
        pnt_paths=point_paths,
        resize_fn=resize_with_padding,
        resize_target_size=input_image_size,
        transform=standard_transform,
        load_multiple_points=True
    )

    test_loader = DataLoader(test_dataset,batch_size=4, num_workers=4,collate_fn=prompt_custom_collate_fn)


    with torch.no_grad():
        total_batches = len(test_loader)  # Get total number of batches
        for batch in tqdm(test_loader, desc=f"Processing {len(test_loader.dataset)} images in {total_batches} batches"):
            images = batch['image'].to(device)               # (B, 3, H, W)
            gt_masks = batch['gt_mask'].to(device)           # (B, 1, H, W)
            prompt_heatmaps = batch['prompt_heatmap'].to(device) # (B, 1, H, W)
            point_classes = batch['point_class'].to(device)      # (B,)
            initial_img_sizes = batch['initial_img_size']
            filenames = batch['img_path']
            # Create target masks based on point class
            target_masks = create_point_wise_mask(
                point_classes, 
                gt_masks
            )

            with torch.no_grad():
                output = model(image=images,
                               prompt_heatmap=prompt_heatmaps,
                               point_class = None)

            # Convert logits to class labels (B, C, H, W) → (B, H, W)
            pred_masks = torch.sigmoid(output) > threshold
            # restore the gt and pred masks shape
            pred_masks = pred_masks.squeeze(1).cpu().numpy().astype(np.uint8)
            target_masks = target_masks.squeeze(1).cpu().numpy().astype(np.uint8)

            resized_pred_masks = [restore_original_mask(pred_masks[i], initial_img_sizes[i][:2]) for i in range(len(images))]
            resized_gt_masks = [restore_original_mask(target_masks[i], initial_img_sizes[i][:2]) for i in range(len(images))]
            # Save prompt gt binary mask
            for i, filename in enumerate(filenames):
                mask_arr = (resized_gt_masks[i]*255).astype(np.uint8)
                mask_save = Image.fromarray(mask_arr, mode='L')
                mask_save_path = os.path.join(gt_save_dir, Path(filename).stem+'.png')  # Keep original filename
                # Ensure parent directory exists
                Path(mask_save_path).parent.mkdir(parents=True, exist_ok=True)
                mask_save.save(mask_save_path,format='PNG')

            # Save predicted masks
            for i, filename in enumerate(filenames):
                pred_arr = (resized_pred_masks[i]*255).astype(np.uint8)
                pred_save = Image.fromarray(pred_arr, mode="L") # Convert the ndarray into RGB Image 
                pred_save_path = os.path.join(pred_save_dir, Path(filename).stem+'.png')  # Keep original filename
                # Ensure parent directory exists
                Path(pred_save_path).parent.mkdir(parents=True, exist_ok=True)
                pred_save.save(pred_save_path,format='PNG')

def prompt_custom_collate_fn(batch):
    # batch is a list of dictionaries, one per sample
    # We'll build batched outputs manually

    images = torch.stack([sample['image'] for sample in batch], dim=0)
    gt_masks = torch.stack([sample['gt_mask'] for sample in batch], dim=0)
    prompt_heatmaps = torch.stack([sample['prompt_heatmap'] for sample in batch], dim=0)
    point_classes = torch.stack([sample['point_class'] for sample in batch], dim=0)
    # Keep `initial_img_size` as a list instead of stacking into a Tensor
    initial_img_sizes = [sample['initial_img_size'] for sample in batch]
    img_paths = [sample['img_path'] for sample in batch]

    return {
        'image': images,
        'gt_mask': gt_masks,
        'prompt_heatmap': prompt_heatmaps,
        'point_class': point_classes,
        'initial_img_size': initial_img_sizes,
        'img_path': img_paths
    }
