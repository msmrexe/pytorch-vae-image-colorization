"""
PyTorch Dataset and DataLoader definitions.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import logging
from .config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS

class BirdDataset(Dataset):
    """Custom Dataset for loading bird images."""
    
    def __init__(self, df, transform=None, grayscale_transform=None):
        self.df = df
        self.transform = transform
        self.grayscale_transform = grayscale_transform
        logging.info(f"BirdDataset initialized with {len(df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_path = self.df.iloc[idx]['filename']
            label = self.df.iloc[idx]['outcome']

            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"Could not read image: {img_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if self.grayscale_transform:
                grayscale_image = self.grayscale_transform(grayscale_image)
            if self.transform:
                rgb_image = self.transform(image)

            # Ensure grayscale has 1 channel
            if len(grayscale_image.shape) == 2:
                 grayscale_image = grayscale_image.unsqueeze(0)
            
            return grayscale_image, rgb_image, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            logging.error(f"Error loading data at index {idx} (path: {img_path}): {e}")
            # Return a dummy tensor or skip
            return self.__getitem__((idx + 1) % len(self)) # Get next item


def get_transforms(img_size=IMG_SIZE):
    """Returns standard transformations for RGB and grayscale images."""
    
    rgb_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    grayscale_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return rgb_transforms, grayscale_transforms

def get_dataloaders(train_df, val_df, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Creates and returns training and validation DataLoaders."""
    
    rgb_transforms, grayscale_transforms = get_transforms()
    
    train_dataset = BirdDataset(
        df=train_df,
        transform=rgb_transforms,
        grayscale_transform=grayscale_transforms
    )
    val_dataset = BirdDataset(
        df=val_df,
        transform=rgb_transforms,
        grayscale_transform=grayscale_transforms
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    logging.info(f"Train DataLoader: {len(train_loader)} batches")
    logging.info(f"Validation DataLoader: {len(val_loader)} batches")
    
    return train_loader, val_loader
