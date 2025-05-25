import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from calculate_IoU import compute_dataset_iou
import json
from tqdm import tqdm

def evaluate_perturbation_robustness(base_test_dir, pred_base_dir, output_dir):
    """
    Evaluate segmentation performance across different perturbation types and levels.
    
    Args:
        base_test_dir: Path to the base directory containing all test datasets
        pred_base_dir: Path to the base directory containing all prediction results
        output_dir: Directory to save evaluation results and plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store results for each perturbation type and level
    results = {}
    
    # Define perturbation types
    perturbation_types = [
        "gaussian_noise_std",
        "gaussian_blur_times",
        "contrast_increase_factor",
        "contrast_decrease_factor",
        "brightness_increase_value",
        "brightness_decrease_value",
        "occlusion_size",
        "salt_pepper_amount"
    ]
    
    # Define level values for x-axis plotting
    level_values = {
        "gaussian_noise_std": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        "gaussian_blur_times": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "contrast_increase_factor": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
        "contrast_decrease_factor": [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
        "brightness_increase_value": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "brightness_decrease_value": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "occlusion_size": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "salt_pepper_amount": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    }
    
    base_test_dir = Path(base_test_dir)
    pred_base_dir = Path(pred_base_dir)
    
    # For each perturbation type
    for pert_type in tqdm(perturbation_types, desc="Perturbation types"):
        results[pert_type] = {
            "levels": [],
            "background_iou": [],
            "cat_iou": [],
            "dog_iou": [],
            "mean_iou": []
        }
        
        pert_dir = base_test_dir / pert_type
        if not pert_dir.exists():
            print(f"Warning: {pert_dir} does not exist. Skipping.")
            continue
            
        # For each perturbation level
        for level in sorted(os.listdir(pert_dir)):
            level_val = level_values[pert_type][int(level)] if pert_type in level_values and int(level) < len(level_values[pert_type]) else level
            
            # Directories
            gt_dir = pert_dir / level / "label"
            pred_dir = pred_base_dir / pert_type / level
            
            if not gt_dir.exists():
                print(f"Warning: GT directory {gt_dir} does not exist. Skipping level {level}.")
                continue
                
            if not pred_dir.exists():
                print(f"Warning: Prediction directory {pred_dir} does not exist. Skipping level {level}.")
                continue
                
            # Output file for this specific test
            iou_output_file = output_dir / f"{pert_type}_{level}_iou.txt"
            
            # Compute IoU
            print(f"\nEvaluating {pert_type} at level {level}...")
            iou_results = compute_dataset_iou(
                gt_folder=gt_dir,
                pred_folder=pred_dir,
                output_file=iou_output_file
            )
            
            # Parse results from the output file
            with open(iou_output_file, 'r') as f:
                content = f.read()
                
            # Extract IoU values using simple parsing
            bg_iou = float(content.split("IoU of Background: ")[1].split("\n")[0])
            cat_iou = float(content.split("IoU of Cats: ")[1].split("\n")[0])
            dog_iou = float(content.split("IoU of Dogs: ")[1].split("\n")[0])
            mean_iou = float(content.split("Mean IoU: ")[1].split("\n")[0])
            
            # Store results
            results[pert_type]["levels"].append(level_val)
            results[pert_type]["background_iou"].append(bg_iou)
            results[pert_type]["cat_iou"].append(cat_iou)
            results[pert_type]["dog_iou"].append(dog_iou)
            results[pert_type]["mean_iou"].append(mean_iou)
    
    # Save all results to a JSON file
    with open(output_dir / "perturbation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    create_plots(results, output_dir)
    
    return results

def create_plots(results, output_dir):
    """Create plots showing IoU vs perturbation level for each perturbation type."""
    
    # Make sure the plots directory exists
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Create individual plots for each perturbation type
    for pert_type, data in results.items():
        if not data["levels"]:
            print(f"No data for {pert_type}, skipping plot")
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(data["levels"], data["background_iou"], 'b-', marker='o', label='Background IoU')
        plt.plot(data["levels"], data["cat_iou"], 'g-', marker='s', label='Cat IoU')
        plt.plot(data["levels"], data["dog_iou"], 'r-', marker='^', label='Dog IoU')
        plt.plot(data["levels"], data["mean_iou"], 'k-', marker='*', linewidth=2, label='Mean IoU')
        
        plt.title(f'Segmentation Performance vs. {pert_type.replace("_", " ").title()}')
        plt.xlabel(pert_type.replace("_", " ").title())
        plt.ylabel('IoU Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_dir / f"{pert_type}_iou_plot.png", dpi=300)
        plt.close()
    
    # Create a summary plot with mean IoU for all perturbation types
    plt.figure(figsize=(12, 8))
    
    for pert_type, data in results.items():
        if not data["levels"]:
            continue
        
        # Normalize the x-axis as percentage of max perturbation
        normalized_levels = np.array(range(len(data["levels"]))) / (len(data["levels"]) - 1 if len(data["levels"]) > 1 else 1)
        plt.plot(normalized_levels, data["mean_iou"], marker='o', label=pert_type.replace("_", " ").title())
    
    plt.title('Mean IoU vs. Perturbation Level (Normalized)')
    plt.xlabel('Perturbation Level (Normalized)')
    plt.ylabel('Mean IoU Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(plot_dir / "all_perturbations_mean_iou.png", dpi=300)
    plt.close()

def extract_iou_from_file(file_path):
    """Extract IoU values from the output file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract IoU values
    bg_iou = float(content.split("IoU of Background: ")[1].split("\n")[0])
    cat_iou = float(content.split("IoU of Cats: ")[1].split("\n")[0])
    dog_iou = float(content.split("IoU of Dogs: ")[1].split("\n")[0])
    mean_iou = float(content.split("Mean IoU: ")[1].split("\n")[0])
    
    return {
        "background_iou": bg_iou,
        "cat_iou": cat_iou,
        "dog_iou": dog_iou,
        "mean_iou": mean_iou
    }

def compute_dataset_iou_with_return(gt_folder, pred_folder, output_file):
    """
    Modified version of compute_dataset_iou that also returns the IoU values.
    """
    compute_dataset_iou(gt_folder, pred_folder, output_file)
    return extract_iou_from_file(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation robustness across perturbations")
    parser.add_argument('--test_dir', type=str, required=True, 
                        help='Path to the base test directory (e.g., ./Dataset/Processed_test)')
    parser.add_argument('--pred_dir', type=str, required=True, 
                        help='Path to the base predictions directory')
    parser.add_argument('--output_dir', type=str, default='./robustness_results',
                        help='Directory to save evaluation results and plots')

    args = parser.parse_args()

    evaluate_perturbation_robustness(
        base_test_dir=args.test_dir,
        pred_base_dir=args.pred_dir,
        output_dir=args.output_dir
    )
    
    print(f"\n✅ Robustness evaluation complete! Results saved in {args.output_dir}")
    print(f"   - View plots in {os.path.join(args.output_dir, 'plots')}")
    print(f"   - Full results in {os.path.join(args.output_dir, 'perturbation_results.json')}")