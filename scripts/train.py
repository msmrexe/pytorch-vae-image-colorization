"""
Main training script for the VAE Colorizer.
"""

import os
import argparse
import torch
import torch.optim as optim
import logging

from src.config import *
from src.utils import setup_logging, load_data_paths, save_loss_plot
from src.dataset import get_dataloaders
from src.vae_system import VAEColorizer
from src.engine import train_model

def main(args):
    # --- Setup ---
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(LOG_FILE)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    try:
        train_df, _ = load_data_paths(args.data_dir, split_name=TRAIN_SPLIT_NAME)
        val_df, _ = load_data_paths(args.data_dir, split_name=VAL_SPLIT_NAME)
    except FileNotFoundError as e:
        logging.error(f"Data directory not found. Did you run setup_dataset.py?")
        logging.error(e)
        return

    train_loader, val_loader = get_dataloaders(
        train_df, 
        val_df, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    # --- Model Initialization ---
    model = VAEColorizer(latent_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # --- Training ---
    train_loss, val_loss = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        epochs=args.epochs, 
        device=device
    )

    # --- Save Artifacts ---
    try:
        torch.save(model.state_dict(), args.model_save_path)
        logging.info(f"Model saved to {args.model_save_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

    save_loss_plot(train_loss, val_loss, f"{args.output_dir}/training_loss_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VAE Colorization Model")
    
    # Paths
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the dataset directory.')
    parser.add_argument('--model_save_path', type=str, default=MODEL_SAVE_PATH, help='Path to save the trained model.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save outputs (plots).')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    parser.add.argument('--learning_rate', type=float, default=LEARNING_RATE, help='Initial learning rate.')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='Dimensionality of the latent space.')
    
    # System
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use (cuda or cpu).')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for DataLoader.')

    args = parser.parse_args()
    main(args)
