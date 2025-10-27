"""
Training and evaluation engine.
Contains the training loop, validation step, and evaluation functions.
"""

import torch
import logging
from tqdm import tqdm
from sklearn.manifold import TSNE

from .utils import save_evaluation_grid, save_tsne_plot

def training_step(model, batch, optimizer, device):
    """Performs a single training step."""
    model.train()
    optimizer.zero_grad()
    
    grayscale_images, rgb_images, _ = batch
    grayscale_images = grayscale_images.to(device)
    rgb_images = rgb_images.to(device)
    
    recon, mu, logvar = model(grayscale_images)
    loss, recon_loss, kld_loss = model.compute_loss(recon, rgb_images, mu, logvar)
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), recon_loss.item(), kld_loss.item()

def validation_step(model, batch, device):
    """Performs a single validation step."""
    model.eval()
    with torch.no_grad():
        grayscale_images, rgb_images, _ = batch
        grayscale_images = grayscale_images.to(device)
        rgb_images = rgb_images.to(device)
        
        recon, mu, logvar = model(grayscale_images)
        loss, recon_loss, kld_loss = model.compute_loss(recon, rgb_images, mu, logvar)
        
    return loss.item(), recon_loss.item(), kld_loss.item()

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    """Main training loop."""
    train_loss_history = []
    val_loss_history = []
    
    logging.info(f"Starting training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        # --- Training ---
        train_losses = []
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch in train_loop:
            loss, recon, kld = training_step(model, batch, optimizer, device)
            train_losses.append(loss)
            train_loop.set_postfix(loss=loss, recon=recon, kld=kld)
            
        train_loss_avg = torch.tensor(train_losses).mean().item()
        train_loss_history.append(train_loss_avg)

        # --- Validation ---
        val_losses = []
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        for batch in val_loop:
            loss, recon, kld = validation_step(model, batch, device)
            val_losses.append(loss)
            val_loop.set_postfix(loss=loss, recon=recon, kld=kld)
            
        val_loss_avg = torch.tensor(val_losses).mean().item()
        val_loss_history.append(val_loss_avg)
        
        scheduler.step()
        
        logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")
        
    logging.info("Training finished.")
    return train_loss_history, val_loss_history

def run_evaluation(model, data_loader, device, output_dir, num_samples, tsne_samples):
    """Runs evaluation: generates image grid and t-SNE plot."""
    model.eval()
    model.to(device)
    logging.info("Running final evaluation...")

    # --- 1. Generate Image Grid ---
    batch = next(iter(data_loader))
    grayscale_images, rgb_images, _ = batch
    grayscale_images = grayscale_images.to(device)
    
    with torch.no_grad():
        recon, _, _ = model(grayscale_images)
        
    save_evaluation_grid(
        grayscale_images[:num_samples],
        rgb_images[:num_samples],
        recon[:num_samples],
        save_path=f"{output_dir}/colorization_results.png",
        num_samples=num_samples
    )

    # --- 2. Generate t-SNE Plot ---
    logging.info(f"Generating t-SNE plot for {tsne_samples} samples...")
    latents = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Encoding for t-SNE"):
            images, _, _ = batch
            images = images.to(device)
            
            mu, _, _ = model.encoder(images)
            latents.append(mu.cpu())
            
            count += images.size(0)
            if count >= tsne_samples:
                break
                
    latents = torch.cat(latents, dim=0)
    latents = latents[:tsne_samples].numpy()
    
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    latents_2d = tsne.fit_transform(latents)
    
    save_tsne_plot(latents_2d, save_path=f"{output_dir}/latent_space_tsne.png")
    logging.info("Evaluation complete.")
