"""
Configuration file for model hyperparameters and global settings.
"""

import torch

# --- Global Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification"
DATA_DIR = "./data/birds-20-species-image-classification"
LOG_FILE = "logs/training.log"
MODEL_SAVE_PATH = "models/vae_colorizer.pth"
OUTPUT_DIR = "outputs"

# --- Data Settings ---
IMG_SIZE = 160
VAL_SPLIT_NAME = "valid" # The dataset has 'train', 'valid', and 'test'
TRAIN_SPLIT_NAME = "train"

# --- Model Hyperparameters ---
BATCH_SIZE = 32
NUM_WORKERS = 4
LATENT_DIM = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
BETA_KLD = 0.001       # Weight for KL Divergence
LAMBDA_PERCEPTUAL = 0.01 # Weight for Perceptual Loss

# --- VGG Perceptual Loss Settings ---
VGG_FEATURE_LAYERS = [0, 5, 10, 19, 28]

# --- Visualization Settings ---
EVAL_NUM_SAMPLES = 10
TSNE_NUM_SAMPLES = 1000
