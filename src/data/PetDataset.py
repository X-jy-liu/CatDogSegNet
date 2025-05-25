import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data.preprocessing import color2class


class PetDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing image-mask pairs for segmentation.

    Args:
        image_paths (list of str): List of file paths to the images.
        mask_paths (list of str): List of file paths to the corresponding masks.
        resize_fn (callable, optional): A function to resize the image & mask.
        resize_target_size (int): the target size of input images to the model. (assuming it's a square image)
        augment_fn (callable, optional): A function to apply data augmentation to the image.
        transform (callable, optional): Function to apply final normalization (e.g., CLIP transform).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], str]: 
            - image: A tensor of shape (C, H, W) representing the normalized image.
            - mask: A tensor of shape (H, W), with values {0,1} or {0,2} (background vs cat or background vs dog).
            - initial_img_size: Tuple (H, W) representing the original image size.
            - img_path: str name of the color image
    """
    def __init__(self, img_paths, msk_paths, resize_fn=None, resize_target_size=None, transform=None):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.resize_fn = resize_fn  # Function to resize images and masks differently
        self.resize_target_size = resize_target_size
        self.transform = transform
        # make sure # of images is equal to # of masks
        assert len(self.img_paths) == len(self.msk_paths), "Mismatch between images and masks."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load and convert image to RGB
        img_path = str(self.img_paths[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        initial_img_size = tuple(img.shape[:2])
        
        # Load and process mask
        msk_path = str(self.msk_paths[idx])
        msk = Image.open(msk_path).convert("RGB")
        msk = np.array(msk)  # Convert to NumPy array

        # Convert color mask to class labels
        msk = color2class(msk)  # Convert mask from RGB to class labels
        
        # Resize if a resizing function is provided
        if self.resize_fn:
            img = self.resize_fn(img, target_size=self.resize_target_size, is_mask=False)
            msk = self.resize_fn(msk, target_size=self.resize_target_size, is_mask=True)

        # Apply transform only to the image (not the mask)
        if self.transform:
            img = self.transform(Image.fromarray(img)) # only apply the normalization to img

        # Convert mask to tensor
        msk = torch.tensor(msk, dtype=torch.long)
        
        return img, msk, initial_img_size, img_path
    
class PetDatasetWithPrompt(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing image-mask pairs with prompt points
    for interactive segmentation tasks. This version loads pre-sampled points from files.

    Each sample includes:
        - the image
        - the ground truth mask
        - a sampled point from the dataset
        - a corresponding heatmap representing the prompt point

    Args:
        img_paths (list of str): 
            List of file paths to the preprocessed images.
        msk_paths (list of str): 
            List of file paths to the corresponding preprocessed masks.
        pnt_paths (list of str):
            List of file paths to the corresponding sampled points text files.
        resize_fn (callable, optional): 
            A function to resize the image and mask to a target size.
            Signature: resize_fn(image_or_mask, target_size, is_mask)
        resize_target_size (int, optional): 
            Target size for resizing the images and masks (assumed square: target_size x target_size).
        transform (callable, optional): 
            A function to apply final image transformation (e.g., normalization) 
            typically for model input preparation.
        load_multiple_points (bool, optional):
            If True, loads all points from the point file and randomly selects one.
            If False, loads just the first point from the file.

    Returns:
        dict: A dictionary containing:
            - 'image' (torch.Tensor): Normalized image tensor of shape (C, H, W)
            - 'gt_mask' (torch.Tensor): Ground truth mask tensor of shape (1, H, W) with class indices
            - 'prompt_heatmap' (torch.Tensor): Heatmap tensor of shape (1, H, W), with 1 at the prompt point
            - 'prompt_point' (torch.Tensor): Coordinates of the sampled prompt point (x, y)
            - 'point_class' (torch.Tensor): Class of the prompt point
            - 'initial_img_size' (tuple): Original size of the image (H, W)
            - 'img_path' (str): Path to the image file
    """
    def __init__(self, img_paths, msk_paths, pnt_paths, resize_fn=None, 
                 resize_target_size=None, transform=None, load_multiple_points=True):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.pnt_paths = pnt_paths
        self.resize_fn = resize_fn
        self.resize_target_size = resize_target_size
        self.transform = transform
        self.load_multiple_points = load_multiple_points
        
        assert len(self.img_paths) == len(self.msk_paths) == len(self.pnt_paths), \
            "Mismatch between images, masks, and point files."

    def __len__(self):
        return len(self.img_paths)

    def load_points_from_file(self, point_path):
        """
        Load pre-sampled points from text file.
        Format: x,y,class per line
        """
        points = []
        with open(point_path, 'r') as f:
            for line in f:
                x, y, cls = map(int, line.strip().split(','))
                points.append((x, y, cls))
        return points

    def generate_gaussian_heatmap(self, H, W, x, y, sigma=10):
        """Returns a heatmap with a 2D Gaussian centered at (x, y)"""
        xs = np.arange(W)
        ys = np.arange(H)
        xs, ys = np.meshgrid(xs, ys)
        g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
        return g.astype(np.float32)

    def __getitem__(self, idx):
        # Load image
        img_path = str(self.img_paths[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img_h,img_w = img.shape[:2]
        initial_img_size = (img_h,img_w)

        # Load and process mask
        msk_path = str(self.msk_paths[idx])
        msk = Image.open(msk_path).convert("RGB")
        msk = np.array(msk)
        msk = color2class(msk)  # Convert RGB mask to label map based on mode

        # Resize if provided
        if self.resize_fn:
            img = self.resize_fn(img, target_size=self.resize_target_size, is_mask=False)
            msk = self.resize_fn(msk, target_size=self.resize_target_size, is_mask=True)

        # Load pre-sampled points
        point_path = str(self.pnt_paths[idx])
        points = self.load_points_from_file(point_path)
        
        # Choose a point
        if self.load_multiple_points and len(points) > 1:
            point_idx = random.randint(0, len(points) - 1)
            x, y, point_class = points[point_idx]
        else:
            x, y, point_class = points[0]  # Just use the first point

        # Create heatmap
        heatmap = self.generate_gaussian_heatmap(msk.shape[0], msk.shape[1], x, y, sigma=10)

        # Apply image transform (normalization)
        if self.transform:
            img = self.transform(Image.fromarray(img))

        # Convert mask and heatmap to torch tensors
        msk = torch.tensor(msk, dtype=torch.long).unsqueeze(0)        # (1, H, W)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return {
            'image': img,                    # (C, H, W)
            'gt_mask': msk,                  # (1, H, W) multi-class
            'prompt_heatmap': heatmap,       # (1, H, W)
            'prompt_point': torch.tensor([x, y], dtype=torch.long),
            'point_class': torch.tensor(point_class, dtype=torch.long),
            'initial_img_size': initial_img_size, # to restore image size in inference
            'img_path': img_path # used in saving files after inference
        }