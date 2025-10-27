# Image Colorization with Variational Autoencoders (VAE)

A deep learning project that implements a Variational Autoencoder (VAE) with a U-Net-like architecture, self-attention, and perceptual loss to colorize grayscale images of birds. This project was developed for the M.S. Machine Learning course to explore generative models and advanced computer vision techniques.

## Features

* **U-Net VAE:** Implements a Variational Autoencoder with an Encoder-Decoder structure featuring skip connections, similar to a U-Net.
* **Advanced Architecture:** Utilizes modern components including Residual Blocks, Squeeze-and-Excitation (SE) blocks, and Self-Attention layers.
* **Hybrid Loss Function:** Combines standard MSE, KL Divergence (for the latent space), and VGG19-based Perceptual Loss to generate sharp, high-fidelity images.
* **Kaggle Integration:** Includes a simple script to download and set up the "Birds 20 Species" dataset directly from Kaggle.
* **Modular & Scripted:** Refactored from a notebook into a clean, modular Python project with separate scripts for training and evaluation.
 
## Core Concepts & Techniques

* **Generative Models:** Variational Autoencoders (VAEs)
* **Representation Learning:** Learning a compressed latent space for complex data.
* **Encoder-Decoder Architecture:** Using a U-Net structure with skip connections to preserve low-level details.
* **Attention Mechanisms:** `SelfAttention` and `SEBlock` to focus on relevant image features.
* **Transfer Learning:** Using a pre-trained VGG19 network to calculate a Perceptual Loss, which better aligns with human visual perception.

---

## How It Works

### 1. Model Architecture & Core Logic

The core of this project is a **Variational Autoencoder (VAE)**, which learns a probabilistic mapping from grayscale images to color images. The architecture is a U-Net-style Encoder-Decoder.

* **Encoder (`src/model.py:Encoder`):**
    * Takes a 1-channel (grayscale) 160x160 image as input.
    * It consists of a stack of `ResidualBlock`s (each containing `SEBlock`s) and `MaxPool2d` layers to downsample the image and extract features.
    * Skip connections are saved at each downsampling stage.
    * A `SelfAttention` layer is applied at the bottleneck to capture global dependencies.
    * Finally, it outputs the `mu` (mean) and `logvar` (log-variance) that define the latent space distribution.

* **Decoder (`src/model.py:Decoder`):**
    * A sample `z` is drawn from the latent space using the reparameterization trick.
    * This sample is passed through a `SelfAttention` layer and upsampled using `ConvTranspose2d` layers.
    * At each upsampling stage, the corresponding skip connection from the encoder is concatenated. This allows the decoder to access low-level features (like edges) from the input, which is crucial for generating sharp images.
    * The final layer is a `Conv2d` followed by a `Sigmoid` function to output a 3-channel (RGB) 160x160 image.

* **Hybrid Loss Function (`src/vae_system.py:VAEColorizer.compute_loss`):**
    The model is trained on a hybrid loss function to balance three objectives:
    1.  **Pixel-wise Reconstruction ($L_{MSE}$):** A standard Mean Squared Error loss between the predicted color image and the actual color image.
    2.  **Perceptual Loss ($L_{perceptual}$):** This loss (from `src/vae_system.py:PerceptualLoss`) feeds the predicted and actual images through a pre-trained VGG19 network. It then computes the L1 loss between the features extracted at different layers. This encourages the model to generate images that are *perceptually* similar to the target, resulting in sharper and more realistic textures.
    3.  **Kullback-Leibler (KL) Divergence ($L_{KLD}$):** The standard VAE loss term that regularizes the latent space, forcing it to approximate a unit Gaussian distribution. This enables the model's generative capabilities.

    The final loss is a weighted sum:
  
    $$L_{total} = L_{Recon} + \beta \cdot L_{KLD}$$
  
    $$L_{Recon} = L_{MSE} + \lambda \cdot L_{perceptual}$$

### 2. Analysis & Results

The model was trained for 50 epochs. The plots show its performance and the quality of the latent space.

* **Training & Validation Loss:** The model converges steadily, with the validation loss tracking closely with the training loss, indicating no significant overfitting.

* **Colorization Examples:** The model successfully learns to colorize the birds. It captures the general color scheme (e.g., blue for jays, yellow/green for others) and applies it plausibly. The perceptual loss helps maintain sharpness, though some color "bleeding" can be observed.

* **Latent Space (t-SNE):** A t-SNE plot visualizes the 128-dimensional latent space in 2D. We can observe some emerging clusters. This suggests the encoder is learning to group images with similar structural features (e.g., bird pose, background texture) together in the latent space, which is a key goal of representation learning.

---

## Project Structure

```
pytorch-vae-image-colorization/
├── .gitignore
├── LICENSE
├── README.md  
├── requirements.txt               # Project dependencies
├── notebooks/
│   └── colorization_demo.ipynb    # Main notebook to run all scripts
├── scripts/
│   ├── setup_dataset.py           # Script to download and setup the Kaggle dataset
│   ├── train.py                   # Main script to train the model
│   └── evaluate.py                # Main script to evaluate the model and get plots
├── logs/                          # Directory for log files (e.g., training.log)
├── models/                        # Directory for saved .pth models
├── outputs/                       # Directory for saved plots (loss, t-SNE, etc.)
└── src/
    ├── __init__.py                # Makes 'src' a Python package
    ├── config.py                  # Stores all hyperparameters and paths
    ├── dataset.py                 # Contains BirdDataset class and DataLoader functions
    ├── model.py                   # Contains Encoder, Decoder, and other nn.Modules
    ├── vae_system.py              # Contains the main VAEColorizer system and PerceptualLoss
    ├── engine.py                  # Contains the training and evaluation loops
    └── utils.py                   # Contains logging, data loading, and plotting helpers
```

----

## How to Use

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/msmrexe/pytorch-vae-image-colorization.git
    cd pytorch-vae-image-colorization
    ```

2. **Setup Environment & Dataset**

    **Install Dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```

    **Download Kaggle Data:**
   
    This project uses the "Birds 20 Species" dataset from Kaggle. The `setup_dataset.py` script will download it for you. You will be prompted for your Kaggle username and API key.
    
    *(You can get your API key by creating a `kaggle.json` file from your Kaggle Account settings page.)*
    
    ```bash
    python setup_dataset.py
    ```
    
    This will download and extract the data into a `./data/` folder.

3. **Run Training**
    
    To train the model, run the `train.py` script. You can adjust hyperparameters using command-line arguments (e.g., `--epochs 100`).
    
    ```bash
    # Train with default settings (50 epochs)
    python train.py
    
    # Train for a different number of epochs
    python train.py --epochs 75
    ```
    
    The script will save the final model to `models/vae_colorizer.pth` and the loss plot to `outputs/training_loss_plot.png`.

4. **Run Evaluation**

    After training, run the `evaluate.py` script to generate the results grid and t-SNE plot using the saved model.
    
    ```bash
    python evaluate.py
    ```
    
    This will save `colorization_results.png` and `latent_space_tsne.png` to the `outputs/` directory.
   
5. **Run via Notebook**
  
    Alternatively, you can run all steps (install, setup, train, eval) sequentially inside the `Colorization_Demo.ipynb` notebook.

-----

## Author

Feel free to connect or reach out if you have any questions\!
 
  * **Maryam Rezaee**
  * **GitHub:** [@msmrexe](https://github.com/msmrexe)
  * **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)
     
-----

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
