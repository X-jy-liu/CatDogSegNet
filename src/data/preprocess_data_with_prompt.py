#!/usr/bin/env python

import numpy as np
from pathlib import Path
from PIL import Image
import torch
import argparse
from tqdm import tqdm

from preprocessing import (
    resize_with_padding,
    color2class,
    augmentor
)

def sample_point_on_mask(mask: np.ndarray, num_points: int) -> list:
    """
    Sample points on the mask based on class distribution.
    
    Args:
        mask (np.ndarray): Segmentation mask
        num_points (int): Number of points to sample
    
    Returns:
        list: Sampled points [(x, y, class)]
    """
    # Convert mask to class mask if not already
    mask = color2class(mask)
    
    # Get unique classes and their indices
    unique_classes = np.unique(mask)
    sampled_points = []
    
    for _ in range(num_points):
        # Weighted sampling based on class presence
        class_weights = []
        class_pixel_lists = []
        
        for cls in unique_classes:
            class_pixels = np.argwhere(mask == cls)
            class_pixel_lists.append(class_pixels)
            class_weights.append(len(class_pixels))
        
        # Normalize weights
        class_weights = np.array(class_weights) / np.sum(class_weights)
        
        # Select a class based on weights
        selected_class_idx = np.random.choice(len(unique_classes), p=class_weights)
        selected_class = unique_classes[selected_class_idx]
        
        # Sample a point from the selected class
        class_pixels = class_pixel_lists[selected_class_idx]
        point_idx = np.random.randint(0, len(class_pixels))
        point = class_pixels[point_idx]
        
        sampled_points.append((point[1], point[0], selected_class))
    
    return sampled_points

def prepare_final_dataset(raw_img_dir, raw_msk_dir,
                           processed_img_dir, processed_msk_dir,
                           base_augmentations, extra_augmentations,
                           target_size, points_per_image=5, mode=0):
    """
    Runs augmentation, resizing, and point sampling, creating the final training dataset.
    
    Args:
        points_per_image (int): Number of points to sample per image
    """
    # Initialize the paths of raw images and raw masks into lists
    raw_img_paths = sorted(Path(raw_img_dir).glob("*.*"))
    raw_msk_paths = sorted(Path(raw_msk_dir).glob("*.*"))
    
    # Create the save directories for processing images and masks
    processed_img_dir = Path(processed_img_dir)
    processed_msk_dir = Path(processed_msk_dir)
    processed_point_dir = processed_img_dir / "points"
    processed_point_dir.mkdir(parents=True, exist_ok=True)
    processed_img_dir.mkdir(parents=True, exist_ok=True)
    processed_msk_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting dataset preparation...")
    for img_path, msk_path in tqdm(zip(raw_img_paths, raw_msk_paths),
                                    total=len(raw_msk_paths),
                                    desc="Final resizing & augmentation",
                                    unit='image'):
        # Extract the names of raw image and mask
        img_name = img_path.stem
        msk_name = msk_path.stem
        
        # Load image and mask
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("RGB"))
        
        # Step 1: Resize with padding
        img_resized = resize_with_padding(img, target_size=target_size, is_mask=False)
        msk_resized = resize_with_padding(msk, target_size=target_size, is_mask=True)

        if mode == 1:
            # Save only 1 point for original image
            points_file_path = processed_point_dir / f"{img_name}_points.txt"
            sampled_point = sample_point_on_mask(msk_resized, num_points=1)
            x, y, cls = sampled_point[0]
            with open(points_file_path, 'w') as f:
                f.write(f"{x},{y},{cls}\n")
            continue  # skip the rest

        # Determine number of augmentations
        msk_class = color2class(msk_resized)
        unique_classes = torch.unique(torch.tensor(msk_class)).tolist()
        num_augmentations = extra_augmentations if 1 in unique_classes else base_augmentations
        
        for i in range(num_augmentations):
            # Augment image and mask
            augmented = augmentor(img_resized, msk_resized)
            aug_image, aug_mask = augmented['image'], augmented['mask']
            # Sample points on the mask
            sampled_points = sample_point_on_mask(aug_mask, num_points=points_per_image)
            # Save augmented image
            aug_img_path = processed_img_dir / f"aug_{img_name}_{i+1}.jpg"
            Image.fromarray(aug_image.astype(np.uint8)).save(aug_img_path)
            
            # Save augmented mask
            aug_msk_path = processed_msk_dir / f"aug_{msk_name}_{i+1}.png"
            Image.fromarray(aug_mask.astype(np.uint8)).save(aug_msk_path)
            
            # Save sampled points for this augmented image
            points_file_path = processed_point_dir / f"aug_{img_name}_{i+1}_points.txt"
            with open(points_file_path, 'w') as f:
                for x, y, cls in sampled_points:
                    f.write(f"{x},{y},{cls}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1],
                        help="0 = full preprocessing (image, mask, prompt), 1 = prompt point only mode")
    parser.add_argument('--img_dir', default="./Dataset/TrainVal/color",
                        help="Directory containing raw images")
    parser.add_argument('--msk_dir', default="./Dataset/TrainVal/label",
                        help="Directory containing raw masks")
    parser.add_argument('--processed_img_dir', default="./Dataset/ProcessedWithPrompt/color",
                        help="Directory to save processed images and points")
    parser.add_argument('--processed_msk_dir', default="./Dataset/ProcessedWithPrompt/label",
                        help="Directory to save processed masks")
    parser.add_argument('--points_per_image', default=5,
                        help="Number of prompt points to generate")
    parser.add_argument('--base_augmentations', default=4,
                        help="Number of base augmentations")
    parser.add_argument('--extra_augmentations', default=8,
                        help="Number of extra augmentations")
    parser.add_argument('--target_size', default=512,
                        help="Target size for resizing images and masks")
    args = parser.parse_args()

    prepare_final_dataset(
        raw_img_dir=Path(args.img_dir),
        raw_msk_dir=Path(args.msk_dir),
        processed_img_dir=Path(args.processed_img_dir),
        processed_msk_dir=Path(args.processed_msk_dir),
        base_augmentations=args.base_augmentations,
        extra_augmentations=args.extra_augmentations,
        target_size=args.target_size,
        points_per_image=args.points_per_image,
        mode=args.mode
    )