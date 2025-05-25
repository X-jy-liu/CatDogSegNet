import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

class SizeDistributionDataset(Dataset):

    def __init__(self, msk_paths):
        self.msk_paths = msk_paths
        # Initialize a dictionary to store pixel counts for each class
        self.over_size = 0

    def __len__(self):
        return len(self.msk_paths)

    def __getitem__(self, idx):
        # Load and process mask
        msk_path = self.msk_paths[idx]
        msk = Image.open(msk_path).convert("RGB")
        msk = np.array(msk)  # Convert to NumPy array
        height, width = msk.shape[:2]
        if height > 512 or width > 512:
            self.over_size += 1
        
        return msk  # Return the mask (this is important for proper dataset behavior)

    def over_size(self):
        return self.over_size

msk_dir = 'Dataset/Processed/label/'
msk_paths = sorted(Path(msk_dir).glob('*.*'))
print(len(msk_paths))

dataset = SizeDistributionDataset(msk_paths)

# Create the DataLoader for batching and iteration
batch_size = 1  # You can adjust this based on your requirements
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Iterate over the DataLoader to process all masks
for _ in tqdm(data_loader,desc='Analysing Mask Classes',unit='image'):
    pass

num_over_size = dataset.over_size
print(f'The number of over sized images is {num_over_size}')