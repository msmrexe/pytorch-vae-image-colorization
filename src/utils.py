"""
Utility functions for logging, data loading, and visualization.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torch

def setup_logging(log_file):
    """Configures the logger to output to console and file."""
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured successfully.")

def load_data_paths(data_dir, split_name='train'):
    """
    Loads image file paths and labels into a pandas DataFrame.
    
    Args:
        data_dir (str): The root directory of the dataset.
        split_name (str): The name of the split to load (e.g., 'train', 'valid').

    Returns:
        pd.DataFrame: A DataFrame with 'filename' and 'outcome' columns.
        dict: A map from class name to integer label.
    """
    filenames = []
    outcomes = []
    class_map = {}
    class_counter = 0
    
    split_path = os.path.join(data_dir, split_name)
    if not os.path.exists(split_path):
        logging.error(f"Dataset split path not found: {split_path}")
        raise FileNotFoundError(f"Directory not found: {split_path}")
        
    logging.info(f"Loading data from: {split_path}")
    
    class_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

    for class_folder in tqdm(class_folders, desc=f"Scanning {split_name} data"):
        class_folder_path = os.path.join(split_path, class_folder)
        
        if class_folder not in class_map:
            class_map[class_folder] = class_counter
            class_counter += 1
            
        for file in os.listdir(class_folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(class_folder_path, file)
                filenames.append(full_path)
                outcomes.append(class_map[class_folder])
                
    df = pd.DataFrame({
        'filename': filenames,
        'outcome': outcomes
    })
    
    df = df.sample(frac=1).reset_index(drop=True)
    logging.info(f"Loaded {len(df)} images for {split_name} split.")
    return df, class_map

def unnormalize(tensor, is_grayscale=False):
    """Un-normalizes a tensor from [-1, 1] to [0, 1]."""
    if is_grayscale:
        mean = torch.tensor([0.5]).view(1, 1, 1)
        std = torch.tensor([0.5]).view(1, 1, 1)
    else:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    
    # Move mean and std to tensor's device
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    return (tensor * std + mean).clamp(0, 1)

def save_loss_plot(train_loss, val_loss, save_path):
    """Saves a plot of training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss plot saved to {save_path}")

def save_evaluation_grid(grayscale_images, actual_images, predicted_images, save_path, num_samples=10):
    """Saves a grid of input, actual, and predicted images."""
    fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        grayscale_img = unnormalize(grayscale_images[i].cpu(), is_grayscale=True).squeeze(0).numpy()
        actual_img = unnormalize(actual_images[i].cpu()).permute(1, 2, 0).numpy()
        pred_img = unnormalize(predicted_images[i].cpu()).permute(1, 2, 0).numpy()

        axes[i, 0].imshow(grayscale_img, cmap='gray')
        axes[i, 0].set_title("Grayscale Input")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(actual_img)
        axes[i, 1].set_title("Actual Image")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_img)
        axes[i, 2].set_title("Predicted Image")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Evaluation image grid saved to {save_path}")

def save_tsne_plot(latents_2d, save_path):
    """Saves a 2D t-SNE scatter plot of latent vectors."""
    plt.figure(figsize=(8, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], s=5, alpha=0.7)
    plt.title("2D t-SNE of Latent Representations")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"t-SNE plot saved to {save_path}")
