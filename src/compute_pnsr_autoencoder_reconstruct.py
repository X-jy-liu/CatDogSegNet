import numpy as np
import torch
from data.preprocessing import resize_with_padding, standard_transform
from pathlib import Path
from tqdm import tqdm
from PIL import Image

img_dir = '/home/s2644572/cv_miniProject2submit/Dataset/Val/color'
img_paths = sorted(Path(img_dir).glob("*.*"))
from models.autoencoder_segmentation import AutoEncoder

model = AutoEncoder()
model.load_state_dict(torch.load('params/train_autoencoder_pretrain_unified_size.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# restore the original image size
def restore_image_size(padded_img: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Restores a padded image of shape (512, 512, 3) to its original shape.
    
    Args:
        padded_img (np.ndarray): The image after padding and resizing, shape (512, 512, 3).
        original_shape (tuple): The original image shape, e.g., (H, W, 3).
    
    Returns:
        np.ndarray: The image cropped to the original shape.
    """
    orig_h, orig_w, _ = original_shape
    padded_h, padded_w, _ = padded_img.shape
    
    # Compute scale factor
    scale = min(512 / orig_h, 512 / orig_w)
    scaled_h = int(orig_h * scale)
    scaled_w = int(orig_w * scale)
    
    # Crop to scaled image region (remove padding)
    top = (padded_h - scaled_h) // 2
    left = (padded_w - scaled_w) // 2
    cropped_img = padded_img[top:top + scaled_h, left:left + scaled_w, :]
    
    # Resize back to original size
    from PIL import Image
    cropped_img = Image.fromarray((cropped_img * 255).astype(np.uint8))  # Ensure [0,255] for PIL
    restored_img = cropped_img.resize((orig_w, orig_h), Image.BICUBIC)
    return np.array(restored_img)

# Function to calculate PSNR
def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original (np.ndarray): The original image.
        reconstructed (np.ndarray): The reconstructed image.
    
    Returns:
        float: The PSNR value in decibels (dB).
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # No noise, perfect reconstruction
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Loop through images and calculate PSNR
psnr_values = []

for img_path in tqdm(img_paths, desc="Processing images", unit='image'):
    # Load and store original shape
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    ini_shape = img_np.shape

    # Preprocess: resize with padding and standardize
    padded_img = resize_with_padding(img_np, target_size=512, is_mask=False)
    tensor_img = standard_transform(padded_img).unsqueeze(0).to(device)

    # Reconstruct using the autoencoder
    with torch.no_grad():
        reconstructed = model(tensor_img).squeeze(0).cpu().numpy()
        reconstructed = np.transpose(reconstructed, (1, 2, 0))  # CxHxW → HxWxC

    # Restore original size
    restored_img = restore_image_size(reconstructed, ini_shape)

    # Convert original image to same format
    original_for_psnr = img_np.astype(np.float32)
    restored_for_psnr = restored_img.astype(np.float32)

    # Calculate PSNR
    psnr = calculate_psnr(original_for_psnr, restored_for_psnr)
    psnr_values.append(psnr)

# Report average
mean_psnr = np.mean(psnr_values)
print(f"\n📊 Average PSNR across {len(psnr_values)} images: {mean_psnr:.2f} dB")
