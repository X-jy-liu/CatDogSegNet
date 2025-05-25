# All preprocessing utilities (image/mask transformations & augmentations)

import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import albumentations as A

# ----------------------------
# Color Mapping Utilities
# ----------------------------

def color2class(img: np.ndarray) -> np.ndarray:
    """
    Convert a label RGB image to pixel-wise class labels.

    Parameters:
        img (np.ndarray): Input image as a NumPy array (expected shape: (H, W, 3)).

    Returns:
        np.ndarray: Output array pixel-wise class labels (expected shape: (H, W)).
    """
    
    color_to_class = {
        (0, 0, 0): 0,         # Black -> Class 0
        (255, 255, 255): 0,   # White -> Class 0
        (128, 0, 0): 1,       # Dark Red -> Class 1
        (0, 128, 0): 2        # Green -> Class 2
    }
    h,w,_ = img.shape
    class_map = np.zeros((h,w), dtype=np.uint8)
    img_reshaped = img.reshape(-1,3)
    class_map_reshaped = np.zeros(img_reshaped.shape[0], dtype=np.uint8)

    # Assign class labels
    for color, class_id in color_to_class.items():
        mask = np.all(img_reshaped == color, axis=1)
        class_map_reshaped[mask] = class_id

    # Reshape back to original dimensions
    class_map = class_map_reshaped.reshape(h, w)

    return class_map

def class2color(class_map: np.ndarray) -> np.ndarray:
    """
    Convert a pixel-wise class label map to an RGB image representation.

    Parameters:
        class_map (np.ndarray): 2D numpy array of shape (H, W) containing class labels.

    Returns:
        np.ndarray: Output RGB image as a NumPy array of shape (H, W, 3).
    """

    # Define the mapping from class labels to RGB colors
    class_to_color = {
        0: (0, 0, 0),         # Class 0 -> Black (background)
        1: (128, 0, 0),       # Class 1 -> Dark Red (cat)
        2: (0, 128, 0)        # Class 2 -> Green (dog)
    }

    # Get image dimensions
    h, w = class_map.shape

    # Initialize an empty RGB image
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign colors based on class labels
    for class_id, color in class_to_color.items():
        mask = (class_map == class_id)
        color_image[mask] = color  # Assign the corresponding RGB color

    return color_image

# ----------------------------
# Resizing Utilities
# ----------------------------

def resize_with_padding(img: np.ndarray, target_size, fill=0, is_mask=False) -> np.ndarray:
    """
    Resize an image while maintaining its aspect ratio and pad it to a square.

    Args:
        img (np.ndarray): Input image (H, W, C) or (H, W) if grayscale/mask.
        target_size (int, optional): The target width and height (default: 224).
        fill (int or tuple, optional): Padding color, either an int (grayscale) or 
            (R, G, B) tuple for color images. Default is black (0).
        is_mask (bool, optional): If True, uses NEAREST interpolation for masks 
            to avoid artifacts. Default is False (for normal images).

    Returns:
        np.ndarray: The resized and padded image/mask with dimensions (target_size, target_size).
    """
    
    # Get current dimensions
    h, w = img.shape[:2]
    
    # Compute scale factor to fit the longest side
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)  # New dimensions

    # Select interpolation method (NEAREST for masks, BICUBIC for images)
    interpolation = cv.INTER_NEAREST if is_mask else cv.INTER_CUBIC

    # Resize image
    img_resized = cv.resize(img, (new_w, new_h), interpolation=interpolation)

    # Create a blank canvas with padding
    if len(img.shape) == 3:  # RGB image
        padded_img = np.full((target_size, target_size, 3), fill, dtype=img.dtype)
    else:  # Grayscale/mask
        padded_img = np.full((target_size, target_size), fill, dtype=img.dtype)

    # Compute padding offsets to center the image
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # Place the resized image onto the padded canvas
    padded_img[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = img_resized

    return padded_img

clip_transform = transforms.Compose([
                    transforms.ToTensor(),  # Converts image to [0, 1] range
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])  # Converts to ~[-1, 1]
                    ])

standard_transform = transforms.Compose([
    transforms.ToTensor()
])

# ----------------------------
# Data Augmentation Utilities
# ---------------------------

def augmentor(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Apply Albumentations-based augmentations to both image and mask.

    Args:
        image (np.ndarray): The input image (H, W, C).
        mask (np.ndarray): The segmentation mask (H, W).

    Returns:
        dict: Dictionary containing:
            - 'image' (np.ndarray): The augmented image (H, W, C).
            - 'mask' (np.ndarray): The augmented mask (H, W).
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # Flip 50% of the time
        A.RandomBrightnessContrast(p=0.2),  # Adjust brightness & contrast
        A.Affine(
            scale=(0.9, 1.1), 
            translate_percent=(-0.0625, 0.0625), 
            rotate=(-15, 15), 
            interpolation=0,  # cv2.INTER_NEAREST (Ensures mask values stay discrete)
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Blur occasionally
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3)  # Color jitter
    ])
    
    # Apply transformations
    augmented = transform(image=image, mask=mask)
    
    return augmented

###########################
# Create Prompt-wise Mask #
###########################
def create_point_wise_mask(point_classes, gt_masks):
    """
    Create a point-wise mask based on the prompt point and its class.
    
    Args:
        prompt_point (torch.Tensor): Tensor of shape (B, 2) with x,y coordinates
        point_class (torch.Tensor): Tensor of shape (B,) with class indices
        gt_mask (torch.Tensor): Ground truth mask of shape (B, 1, H, W)
        
    Returns:
        torch.Tensor: Binary mask showing regions that should belong to the same class
                     as the prompt point
    """
    B, _, H, W = gt_masks.shape
    point_wise_masks = []
    
    for b in range(B):
        cls = point_classes[b]
        
        # Create binary mask where 1s represent pixels of the same class as the prompt point
        point_wise_mask = (gt_masks[b, 0] == cls).float()
        
        point_wise_masks.append(point_wise_mask)

    return torch.stack(point_wise_masks).unsqueeze(1).long()  # (B, 1, H, W)