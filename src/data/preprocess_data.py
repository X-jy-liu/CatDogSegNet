# src/preprocess_dataset.py

from preprocessing import (
    resize_with_padding, 
    color2class, 
    augmentor
)
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import argparse

def prepare_final_dataset(raw_img_dir, raw_msk_dir, 
                          processed_img_dir, processed_msk_dir, 
                          base_augmentations, extra_augmentations,
                          target_size):
    """
    Runs augmentation and resizing, creating the final training dataset.
    """

    # Step 0: Initialize the paths of raw images and raw masks into lists
    raw_img_paths = sorted(Path(raw_img_dir).glob("*.*"))
    print(len(raw_img_paths))
    raw_msk_paths = sorted(Path(raw_msk_dir).glob("*.*"))
    
    # create the save directories for processing images and masks
    processed_img_dir.mkdir(parents=True, exist_ok=True)
    processed_msk_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting dataset preparation...")
    for img_path, msk_path in tqdm(zip(raw_img_paths, raw_msk_paths), 
                                   total=len(raw_msk_paths), 
                                   desc="Final resizing & augmentation",
                                   unit= 'image'):
        # Extract the names of raw image and mask
        img_name = img_path.stem
        msk_name = msk_path.stem
        # Load image and mask
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("RGB"))

        # Step 1: Resize with padding
        img_resized = resize_with_padding(img, target_size=target_size, is_mask=False)
        msk_resized = resize_with_padding(msk, target_size=target_size, is_mask=True)

        msk_class = color2class(msk_resized)
        unique_classes = torch.unique(torch.tensor(msk_class)).tolist()
        num_augmentations = extra_augmentations if 1 in unique_classes else base_augmentations
        for i in range(num_augmentations):
            augmented = augmentor(img_resized, msk_resized)
            aug_image, aug_mask = augmented['image'], augmented['mask']

            Image.fromarray(aug_image.astype(np.uint8)).save(
                Path(processed_img_dir) / f"aug_{img_name}_{i+1}.jpg")

            Image.fromarray(aug_mask.astype(np.uint8)).save(
                Path(processed_msk_dir) / f"aug_{msk_name}_{i+1}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the final dataset for segmentation training.")
    parser.add_argument('--img_dir', default="./Dataset/Train/color",
                        help="Directory containing raw images")
    parser.add_argument('--msk_dir', default="./Dataset/Train/label",
                        help="Directory containing raw masks")
    parser.add_argument('--processed_img_dir', default="./Dataset/TrainProcessed/color",
                        help="Directory to save processed images and points")
    parser.add_argument('--processed_msk_dir', default="./Dataset/TrainProcessed/label",
                        help="Directory to save processed masks")
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
        target_size=args.target_size
    )
