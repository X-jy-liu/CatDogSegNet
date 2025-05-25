import os
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import scipy.ndimage as ndimage
from skimage.util import random_noise
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from data.preprocessing import resize_with_padding, color2class, class2color

def create_test_datasets(raw_img_dir, raw_msk_dir, output_base_dir, target_size=512, 
                        pert_types=None, num_workers=4):
    output_dir = Path(output_base_dir) / "Processed_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_perturbations = {
        "gaussian_noise_std": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        "gaussian_blur_times": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "contrast_increase_factor": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
        "contrast_decrease_factor": [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
        "brightness_increase_value": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "brightness_decrease_value": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "occlusion_size": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "salt_pepper_amount": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    }

    if pert_types:
        perturbations = {k: all_perturbations[k] for k in pert_types if k in all_perturbations}
    else:
        perturbations = all_perturbations

    #img_paths = sorted(Path(raw_img_dir).glob("*.*"))
    valid_exts = {".png", ".jpg", ".jpeg"}
    img_paths = sorted([p for p in Path(raw_img_dir).glob("*") 
                    if p.suffix.lower() in valid_exts and not p.name.startswith(".")])

    msk_paths = sorted([p for p in Path(raw_msk_dir).glob("*") 
                    if p.suffix.lower() in valid_exts and not p.name.startswith(".")])


    print(f"Found {len(img_paths)} test images and {len(msk_paths)} test masks")
    print(f"Creating test datasets for perturbation types: {list(perturbations.keys())}")

    for pert_type, levels in perturbations.items():
        pert_dir = output_dir / pert_type
        pert_dir.mkdir(exist_ok=True)

        for level in levels:
            level_dir = pert_dir / str(level)
            level_dir.mkdir(exist_ok=True)

            img_dir = level_dir / "color"
            msk_dir = level_dir / "label"
            img_dir.mkdir(exist_ok=True)
            msk_dir.mkdir(exist_ok=True)

    for pert_type, levels in perturbations.items():
        print(f"\nProcessing perturbation type: {pert_type}")

        tasks = []
        for i, (img_path, msk_path) in enumerate(zip(img_paths, msk_paths)):
            tasks.append((img_path, msk_path, pert_type, levels, output_dir, target_size))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_image_for_perturbation, *task) for task in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {pert_type}"):
                pass

def process_image_for_perturbation(img_path, msk_path, pert_type, levels, output_dir, target_size):
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("RGB"))

        img_resized = resize_with_padding(img, target_size=target_size, is_mask=False)
        msk_resized = resize_with_padding(msk, target_size=target_size, is_mask=True)

        for level in levels:
            level_img_dir = output_dir / pert_type / str(level) / "color"
            level_msk_dir = output_dir / pert_type / str(level) / "label"

            out_img_path = level_img_dir / f"{img_path.stem}.jpg"
            out_msk_path = level_msk_dir / f"{msk_path.stem}.png"

            if out_img_path.exists() and out_msk_path.exists():
                continue

            perturbed_img = apply_perturbation(img_resized, pert_type, level)

            Image.fromarray(perturbed_img.astype(np.uint8)).save(out_img_path)
            Image.fromarray(msk_resized.astype(np.uint8)).save(out_msk_path)

    except Exception as e:
        print(f"Error processing {img_path} for {pert_type}: {e}")



def apply_perturbation(image, pert_type, level):
    if pert_type == "gaussian_noise_std":
        return apply_gaussian_noise(image, level)
    elif pert_type == "gaussian_blur_times":
        return apply_gaussian_blur(image, level)
    elif pert_type == "contrast_increase_factor":
        return apply_contrast_increase(image, level)
    elif pert_type == "contrast_decrease_factor":
        return apply_contrast_decrease(image, level)
    elif pert_type == "brightness_increase_value":
        return apply_brightness_increase(image, level)
    elif pert_type == "brightness_decrease_value":
        return apply_brightness_decrease(image, level)
    elif pert_type == "occlusion_size":
        return apply_occlusion(image, level)
    elif pert_type == "salt_pepper_amount":
        return apply_salt_pepper(image, level)
    else:
        return image

def apply_gaussian_noise(image, std):
    if std == 0:
        return image
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_gaussian_blur(image, blur_times):
    if blur_times == 0:
        return image
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    blurred = image.copy()
    for _ in range(blur_times):
        for i in range(3):
            blurred[:,:,i] = ndimage.convolve(blurred[:,:,i], kernel, mode='reflect')
    return blurred.astype(np.uint8)

def apply_contrast_increase(image, factor):
    if factor == 1.0:
        return image
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def apply_contrast_decrease(image, factor):
    if factor == 1.0:
        return image
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def apply_brightness_increase(image, value):
    if value == 0:
        return image
    brightened = image.astype(np.int16) + value
    return np.clip(brightened, 0, 255).astype(np.uint8)

def apply_brightness_decrease(image, value):
    if value == 0:
        return image
    darkened = image.astype(np.int16) - value
    return np.clip(darkened, 0, 255).astype(np.uint8)

def apply_occlusion(image, size):
    if size == 0:
        return image
    occluded = image.copy()
    h, w = image.shape[:2]
    size = min(size, h, w)
    top = np.random.randint(0, h - size + 1)
    left = np.random.randint(0, w - size + 1)
    occluded[top:top+size, left:left+size, :] = 0
    return occluded

def apply_salt_pepper(image, amount):
    if amount == 0:
        return image
    noisy = random_noise(image, mode='s&p', amount=amount)
    return (noisy * 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test datasets with various perturbations")
    parser.add_argument("--img_dir", type=str, default="./Dataset/Test/color", help="Directory containing test images")
    parser.add_argument("--msk_dir", type=str, default="./Dataset/Test/label", help="Directory containing test masks")
    parser.add_argument("--output_dir", type=str, default="./Dataset", help="Base directory for saving test datasets")
    parser.add_argument("--target_size", type=int, default=512, help="Size to resize images to")
    parser.add_argument("--pert_types", type=str, nargs="+", choices=[
        "gaussian_noise_std", "gaussian_blur_times", "contrast_increase_factor", "contrast_decrease_factor",
        "brightness_increase_value", "brightness_decrease_value", "occlusion_size", "salt_pepper_amount"
    ], help="Specific perturbation types to process (default: all)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")
    args = parser.parse_args()

    create_test_datasets(
        raw_img_dir=Path(args.img_dir),
        raw_msk_dir=Path(args.msk_dir),
        output_base_dir=Path(args.output_dir),
        target_size=args.target_size,
        pert_types=args.pert_types,
        num_workers=args.num_workers
    )
