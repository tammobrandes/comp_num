import os
import pandas as pd
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
import torch
import tempfile

class MultiTaskDataset(Dataset):
    def __init__(self, zip_file, subfolder_name, csv_filename, transform=None):
        print("Initializing MultiTaskDataset...")

        # Extract the ZIP file contents to a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir.name)

        # Set paths to the subfolder and CSV file
        self.subfolder_path = os.path.join(self.temp_dir.name, subfolder_name)
        csv_path = os.path.join(self.subfolder_path, csv_filename)

        # Load the CSV file
        self.annotations = pd.read_csv(csv_path, header=None)  # Adjust header if necessary
        print(f"Size of CSV dataset: {len(self.annotations)} rows")

        # Create label-to-index mappings for class1
        self.class1_labels = self.annotations.iloc[:, 1].unique()
        self.class1_mapping = {label: idx for idx, label in enumerate(self.class1_labels)}
        print(f"Class 1 mapping: {self.class1_mapping}")

        # Numerosity classes for task2
        self.numerosity_classes = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        self.class2_mapping = {num: idx for idx, num in enumerate(self.numerosity_classes)}
        print(f"Class 2 mapping: {self.class2_mapping}")

        # Set the transform
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the image filename from the CSV
        img_name = self.annotations.iloc[idx, 0]
        img_name = img_name.lstrip('/') # requires strip otherwise ignores preceding path
        path_sub =  (self.subfolder_path)
        path_sub_norm = path_sub.replace(os.sep, '/') 
        img_path = os.path.join(path_sub_norm, img_name)

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Map class1 (string) to integer index
        class1_str = self.annotations.iloc[idx, 1]
        class1 = self.class1_mapping[class1_str]

        # Map numerosity class2 (numerosity value) to its corresponding index
        class2_value = int(self.annotations.iloc[idx, 2])
        class2 = self.class2_mapping[class2_value]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, class1, class2

    def __del__(self):
        # Clean up the temporary directory when the dataset is deleted
        self.temp_dir.cleanup()


def get_dataloaders(zip_file, subfolder_name, csv_filename, batch_size=32, seed=42):
    print("Creating DataLoaders...")

    # Set random seed
    torch.manual_seed(seed)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create the dataset
    dataset = MultiTaskDataset(
        zip_file=zip_file, 
        subfolder_name=subfolder_name,
        csv_filename=csv_filename, 
        transform=transform
    )

    # Total dataset size
    total_size = len(dataset)

    # Split dataset
    train_size = 12800
    val_size = 3200
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total dataset size: {total_size}")
    print(f"Train set size: {train_size}, Validation set size: {val_size}, Test set size: {test_size}")

    return train_loader, val_loader, test_loader, test_dataset
