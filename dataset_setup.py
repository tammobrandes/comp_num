## dataset_setup.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class MultiTaskDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        print("Initializing MultiTaskDataset...")
        
        # Load the CSV file (with no header, since the first row is data)
        self.annotations = pd.read_csv(csv_file, header=None)
        print(f"Size of CSV dataset: {len(self.annotations)} rows")

        # Create label-to-index mappings for class1 only
        self.class1_labels = self.annotations.iloc[:, 1].unique()

        # Create a mapping from class1 strings to integer indices
        self.class1_mapping = {label: idx for idx, label in enumerate(self.class1_labels)}
        print(f"Class 1 mapping: {self.class1_mapping}")

        # Numerosity classes for task2
        self.numerosity_classes = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

        # Create a mapping from numerosity values to integer indices
        self.class2_mapping = {num: idx for idx, num in enumerate(self.numerosity_classes)}
        print(f"Class 2 mapping: {self.class2_mapping}")

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the relative path from the CSV and remove any leading slash
        relative_img_path = self.annotations.iloc[idx, 0].lstrip('/')  # Remove leading '/'
        img_name = os.path.join(self.img_dir, relative_img_path)  # Properly join paths
        image = Image.open(img_name).convert("RGB")
        
        # Map class1 (string) to integer index
        class1_str = self.annotations.iloc[idx, 1]
        class1 = self.class1_mapping[class1_str]

        # Map numerosity class2 (numerosity value) to its corresponding index
        class2_value = int(self.annotations.iloc[idx, 2])  # Assuming numerosity is stored as int in the CSV
        class2 = self.class2_mapping[class2_value]

        # Apply any transformations if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and the two class labels
        return image, class1, class2

def get_dataloaders(csv_file, img_dir, batch_size=32, seed=42):
    print("Creating DataLoaders...")
    
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create the dataset
    dataset = MultiTaskDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    # Total number of samples
    total_size = len(dataset)

    # Calculate sizes for training, validation, and test sets
    # if OOD Dataset --> both 0
    # if not OOD --> train = 12800, val = 3200
    train_size = 12800
    val_size = 3200
    test_size = total_size - train_size - val_size  # Ensure the sum equals the total dataset size

    print(f"Total dataset size: {total_size}")
    print(f"Train set size: {train_size}, Validation set size: {val_size}, Test set size: {test_size}")

    # Create a generator with the specified seed for reproducibility
    # uncomment if using non OOD
    
    generator = torch.Generator().manual_seed(seed)

    # Split the dataset using the generator for reproducibility
    # uncomment if using non OOD
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Create DataLoaders
    # Set Train and Val Loader to 0 if using OOD dataset (has not train, val)
    # if non OOD, uncomment DataLoader functions, comment 0 functions; also for test_loader add "test_" to "dataset"
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   # train_loader = 0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #val_loader = 0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
