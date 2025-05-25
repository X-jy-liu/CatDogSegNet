import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from data.preprocessing import color2class
import torch

def binarize_mask(mask_rgb):
    """
    Converts an RGB mask to binary (1: foreground, 0: background)
    Args:
        mask_rgb (np.ndarray): The RGB mask image.
    Returns:
        np.ndarray: Binary mask where 1 represents the foreground and 0 the background.
    """
    gray = np.array(Image.fromarray(mask_rgb).convert("L"))
    binary = (gray > 127).astype(np.uint8)
    return binary

def compute_dataset_iou(mode:str, gt_folder:str, pred_folder:str, output_file:str) -> None:
    '''
    Computes the Intersection over Union (IoU) for a dataset of segmentation masks.
    Args:
        mode (str): Mode of evaluation, either "3-class" or "binary".
        gt_folder (str): Path to the ground truth masks folder.
        pred_folder (str): Path to the predicted masks folder.
        output_file (str): Path to save the IoU results.
    Returns:
        None. (results saved to <output_file>.txt)
    '''
    print(f'Calculating IoU of predicted images in {pred_folder} at mode {mode}...')    
    
    # Define dataset paths
    gt_paths = sorted(Path(gt_folder).glob("*.*"))
    pred_paths = sorted(Path(pred_folder).glob("*.*"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize accumulators
    if mode == "3-class":
        intersection_sum = {0: 0, 1: 0, 2: 0}
        union_sum = {0: 0, 1: 0, 2: 0}
    else:  # binary
        intersection_sum = {1: 0}  # foreground only
        union_sum = {1: 0}

    # Open file in write mode
    with open(output_file, "w") as file:
        for idx in tqdm(range(len(pred_paths)), desc='Test Image Loading', unit='image'):
            if mode == "3-class":
                gt_mask = Image.open(gt_paths[idx]).convert('RGB')
                gt_mask = np.array(gt_mask)
                gt_classes = color2class(gt_mask)
                gt_classes = torch.from_numpy(gt_classes).to(device).long()

                pred_mask = Image.open(pred_paths[idx]).convert('RGB')
                pred_mask = np.array(pred_mask)
                pred_classes = color2class(pred_mask)
                pred_classes = torch.from_numpy(pred_classes).to(device).long()

                # convert pred and gt images into one-hot embedding
                pred_one_hot = torch.nn.functional.one_hot(pred_classes, num_classes=3).permute(2,0,1).float()
                gt_one_hot = torch.nn.functional.one_hot(gt_classes, num_classes=3).permute(2, 0, 1).float()
                intersection = (pred_one_hot * gt_one_hot).sum(dim=(1, 2))
                union = pred_one_hot.sum(dim=(1, 2)) + gt_one_hot.sum(dim=(1, 2)) - intersection

                for cls in range(3):
                    intersection_sum[cls] += intersection[cls].item()
                    union_sum[cls] += union[cls].item()
            # For binary mode, we only care about the foreground (1) and background (0)
            elif mode == "binary":
                gt_mask = Image.open(gt_paths[idx]).convert('RGB')
                pred_mask = Image.open(pred_paths[idx]).convert('RGB')

                gt_binary = binarize_mask(np.array(gt_mask))
                pred_binary = binarize_mask(np.array(pred_mask))

                gt_tensor = torch.from_numpy(gt_binary).to(device)
                pred_tensor = torch.from_numpy(pred_binary).to(device)

                intersection = torch.logical_and(pred_tensor, gt_tensor).sum().item()
                union = torch.logical_or(pred_tensor, gt_tensor).sum().item()

                intersection_sum[1] += intersection
                union_sum[1] += union
        # initialize final iou dict
        final_iou = {}
        # Compute IoU per class
        if mode == "3-class":
            for cls in range(3):
                if union_sum[cls] > 0:
                    final_iou[cls] = intersection_sum[cls] / union_sum[cls]
                else:
                    final_iou[cls] = None  # Optional: mark as N/A if class never appears

            # Mean IoU: Only include classes that exist
            valid_ious = [iou for iou in final_iou.values() if iou is not None]
            mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0

            # Final results
            final_results = (f"\nFinal Dataset-level IoU Results:\n"
                            f"IoU of Background: {final_iou[0]:.4f}\n"
                            f"IoU of Cats: {final_iou[1]:.4f}\n"
                            f"IoU of Dogs: {final_iou[2]:.4f}\n"
                            f"Mean IoU: {mean_iou:.4f}\n")

        elif mode == "binary":
            if union_sum[1] > 0:
                iou = intersection_sum[1] / union_sum[1]
            else:
                iou = 0.0

            final_results = (f"\nFinal Dataset-level IoU Results (Binary):\n"
                             f"IoU of Foreground vs Background: {iou:.4f}\n")
        else:
            raise ValueError("Invalid mode. Choose '3-class' or 'binary'.")
        # Save results to file
        print(final_results.strip())
        file.write(final_results)

    print(f"\nDataset-level IoU results saved in: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Dataset-level IoU for Segmentation Masks")
    parser.add_argument('--mode', type=str, choices=["3-class", "binary"], default="3-class",
                        help='Evaluation mode: 3-class (default) or binary (SAM-style fg/bg)')
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to the ground truth masks folder')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the predicted masks folder')
    parser.add_argument('--output_file', type=str, default='iou_results.txt', help='Path to save the IoU results')

    args = parser.parse_args()
    compute_dataset_iou(args.mode, args.gt_folder, args.pred_folder, args.output_file)