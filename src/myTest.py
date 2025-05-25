import matplotlib.pyplot as plt
import numpy as np
import torch
from data.preprocessing import resize_with_padding, standard_transform
from pathlib import Path
from data.PetDataset import PetDatasetWithPrompt

img_dir = '/home/s2644572/cv_miniProject2submit/Dataset/ProcessedWithPrompt/color'
msk_dir = '/home/s2644572/cv_miniProject2submit/Dataset/ProcessedWithPrompt/label'
pnt_dir = '/home/s2644572/cv_miniProject2submit/Dataset/ProcessedWithPrompt/color/points'
img_paths = sorted(Path(img_dir).glob("*.*"))
msk_paths = sorted(Path(msk_dir).glob("*.*"))
pnt_paths = sorted(Path(pnt_dir).glob('*.*'))

dataset = PetDatasetWithPrompt(img_paths,msk_paths,pnt_paths)

from models.unet_segmentation import UNet
from models.autoencoder_segmentation import AutoEncoderSegmentation, AutoEncoder, AutoEncoder

import numpy as np

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

model = AutoEncoder()
model.load_state_dict(torch.load('params/train_autoencoder_pretrain_unified_size.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

img_path = "/home/s2644572/cv_miniProject2submit/Dataset/Val/color/yorkshire_terrier_189.jpg"

from PIL import Image
img = Image.open(img_path).convert("RGB")
img_arr = np.array(img)
ini_img_shape = img_arr.shape
img = resize_with_padding(img_arr, target_size=512, is_mask=False)
img = standard_transform(img)
img = img.unsqueeze(0).to(device)
print(img.shape)

recontructed = model(img)
recontructed = recontructed.squeeze(0).cpu().detach().numpy()
recontructed = np.transpose(recontructed, (1, 2, 0))
recontructed = restore_image_size(recontructed, ini_img_shape)
print(f'initial image size: {recontructed.shape}')
import matplotlib.pyplot as plt
ax, fig = plt.subplots(1, 2, figsize=(12, 6))
# Display the original and reconstructed images
plt.subplot(1, 2, 1)
plt.imshow(img_arr)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(recontructed)
plt.title("Reconstructed Image")
plt.axis('off')
plt.show()

# save the reconstructed image
output_path = 'reconstructed_image.jpg'

plt.imsave(output_path, recontructed)
print(f"Reconstructed image saved to {output_path}")

# save the original image
output_path = 'original_image.jpg'
plt.imsave(output_path, img_arr)
print(f"Original image saved to {output_path}")

# # calculate the PSNR
# def calculate_psnr(original, reconstructed):
#     mse = np.mean((original - reconstructed) ** 2)
#     if mse == 0:
#         return float('inf')
#     max_pixel = 255.0
#     psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
#     return psnr
# psnr_value = calculate_psnr(img_arr, recontructed)
# print(f"PSNR: {psnr_value:.2f} dB")

# # calculate the mse
# def calculate_mse(original, reconstructed):
#     mse = np.mean((original - reconstructed) ** 2)
#     return mse
# mse_value = calculate_mse(img_arr/255, recontructed/255)
# print(f"MSE: {mse_value:.6f}")

# def compare_pixel_ranges(img1: np.ndarray, img2: np.ndarray):
#     min1, max1 = img1.min(), img1.max()
#     min2, max2 = img2.min(), img2.max()
    
#     print(f"Image 1 range: min={min1:.4f}, max={max1:.4f}")
#     print(f"Image 2 range: min={min2:.4f}, max={max2:.4f}")
    
#     if np.isclose(min1, min2) and np.isclose(max1, max2):
#         print("✅ Pixel value ranges match.")
#     else:
#         print("❌ Pixel value ranges differ.")

# # Example usage:
# compare_pixel_ranges(img_arr, recontructed)