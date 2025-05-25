import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

class ClassDistributionDataset(Dataset):

    def __init__(self, msk_paths):
        self.msk_paths = msk_paths
        # Initialize a dictionary to store pixel counts for each class
        self.class_pixel_count = defaultdict(int)
        self.cat_num = 0
        self.dog_num = 0

    def __len__(self):
        return len(self.msk_paths)

    def __getitem__(self, idx):
        # Load and process mask
        msk_path = self.msk_paths[idx]
        msk = Image.open(msk_path).convert("RGB")
        msk = np.array(msk)  # Convert to NumPy array
        unique_colors = np.unique(msk.reshape(-1,3),axis=0)
        red = np.array([128,0,0])
        green = np.array([0,128,0])
        has_red = np.any(np.all(unique_colors == red, axis=1))
        has_green = np.any(np.all(unique_colors == green, axis=1))

        if has_red and not has_green:
            self.cat_num += 1
        elif not has_red and has_green:
            self.dog_num += 1
        else:
            print(f'Error in {msk_path.name}: both or neither class found')
                
        color_to_class = {
            (0, 0, 0): 0,         # Black -> Class 0
            (128, 0, 0): 1,       # Dark Red -> Class 1
            (0, 128, 0): 2,       # Green -> Class 2
            (255, 255, 255): 0    # White -> Class 3
        }

        # Count pixels for each class
        for color, class_id in color_to_class.items():
            # Count occurrences of each color in the mask
            class_pixels = np.sum(np.all(msk == color, axis=-1))
            # Accumulate the pixel count for the corresponding class
            self.class_pixel_count[class_id] += class_pixels
        
        return msk  # Return the mask (this is important for proper dataset behavior)

    def get_class_pixel_counts(self):
        return self.class_pixel_count

msk_dir = 'Dataset/Processed/label/'
msk_paths = sorted(Path(msk_dir).glob('*.*'))
print(len(msk_paths))

dataset = ClassDistributionDataset(msk_paths)

# Create the DataLoader for batching and iteration
batch_size = 1  # You can adjust this based on your requirements
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Iterate over the DataLoader to process all masks
for _ in tqdm(data_loader,desc='Analysing Mask Classes',unit='image'):
    pass

# Retrieve the pixel counts for each class
pixel_counts = dataset.get_class_pixel_counts()
print(pixel_counts)
print(f"Total Cat Masks: {dataset.cat_num}")
print(f"Total Dog Masks: {dataset.dog_num}")

# Class labels
class_labels = ['Background', 'Cat', 'Dog']

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(pixel_counts.values(), labels=class_labels, autopct='%1.1f%%', startangle=90)
plt.title("Pixel Distribution per Class")
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()