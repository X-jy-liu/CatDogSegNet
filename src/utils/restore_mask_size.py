import cv2 as cv
import numpy as np

def restore_original_mask(mask: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    Resize the class mask back to the original image size.

    Args:
        mask (np.ndarray): The resized mask of shape (target_size, target_size).
        original_size (tuple): The original image dimensions (H, W).

    Returns:
        np.ndarray: The restored mask with original dimensions (H, W).
    """
    target_size = mask.shape[0]  # Assuming square mask (224, 224)
    orig_h, orig_w = original_size

    # Compute scale factor (same used in resize_with_padding)
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Compute padding offsets
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # Crop the valid region (remove padding)
    cropped_mask = mask[paste_y:paste_y + new_h, paste_x:paste_x + new_w]

    # Resize back to original dimensions using NEAREST to keep discrete labels
    original_mask = cv.resize(cropped_mask, (orig_w, orig_h), interpolation=cv.INTER_NEAREST)

    return original_mask