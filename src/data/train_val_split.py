from pathlib import Path

source_color_dir = Path("Dataset/TrainVal/color/")
source_lable_dir = Path("Dataset/TrainVal/label/")

import random
from pathlib import Path
import shutil

# Set seed for reproducibility
random.seed(42)

# Source directories
source_color_dir = Path("Dataset/TrainVal/color/")
source_label_dir = Path("Dataset/TrainVal/label/")

# Destination directories
train_color_dir = Path("Dataset/Train/color/")
train_label_dir = Path("Dataset/Train/label/")
val_color_dir = Path("Dataset/Val/color/")
val_label_dir = Path("Dataset/Val/label/")

# Create destination folders
for path in [train_color_dir, train_label_dir, val_color_dir, val_label_dir]:
    path.mkdir(parents=True, exist_ok=True)

# Get list of color image files (assuming same names for labels)
color_files = sorted(source_color_dir.glob("*"))
total = len(color_files)
split_idx = int(total * 0.9)

# Shuffle and split
random.shuffle(color_files)
train_files = color_files[:split_idx]
val_files = color_files[split_idx:]

# Helper to copy files
def copy_files(file_list, color_dest, label_dest):
    for color_file in file_list:
        label_filename = color_file.stem + ".png"
        label_file = source_label_dir / label_filename
        shutil.copy2(color_file, color_dest / color_file.name)
        shutil.copy2(label_file, label_dest / label_file.name)

# Copy files to train and val directories
copy_files(train_files, train_color_dir, train_label_dir)
copy_files(val_files, val_color_dir, val_label_dir)

print(f"Total images: {total}")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")
