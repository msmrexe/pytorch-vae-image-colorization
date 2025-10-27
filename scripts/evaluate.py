"""
Main evaluation script.
Loads a trained model and generates:
1. A grid of (Grayscale, Actual, Predicted) images.
2. A t-SNE plot of the latent space.
"""

import os
import argparse
import torch
import logging

# --- Add project root to sys.path ---
# This allows imports from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------

from src.config import *
from src.utils import setup_logging, load_data_paths
from src.dataset import get_dataloaders
from src.vae_system import VAEColorizer
from src.engine import run_evaluation

def main(args):
    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging("logs/evaluation.log")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    try:
        # We use the validation data for evaluation
        val_df, _ = load_data_paths(args.data_dir, split_name=VAL_SPLIT_NAME)
    except FileNotFoundError as e:
        logging.error(f"Data directory not found: {args.data_dir}")
        logging.error(e)
        return

    # Need to get a dataloader for evaluation
    # Batch size can be the same, shuffle=False
    _, val_loader = get_dataloaders(
        val_df, # Pass val_df as dummy for train_df
        val_df, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    # --- Model Loading ---
    try:
        model = VAEColorizer(latent_dim=args.latent_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found: {args.model_path}. Please train the model first.")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # --- Run Evaluation ---
    run_evaluation(
        model,
        val_loader,
        device,
        args.output_dir,
        num_samples=args.num_samples,
        tsne_samples=args.tsne_samples
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the VAE Colorization Model")
    
    # Paths
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the dataset directory.')
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH, help='Path to the saved trained model.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save evaluation plots.')
    
    # Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='Latent dim of the loaded model.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for evaluation.')
    
    # System
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use (cuda or cpu).')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for DataLoader.')
    
    # Evaluation settings
    parser.add_argument('--num_samples', type=int, default=EVAL_NUM_SAMPLES, help='Number of images to show in the results grid.')
    parser.add_argument('--tsne_samples', type=int, default=TSNE_NUM_SAMPLES, help='Number of samples for t-SNE plot.')

    args = parser.parse_args()
    main(args)
