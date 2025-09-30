# src/climate_vae.py
# Variational AutoEncoder for Climate Representation Learning
# Implements VAE for learning latent climate patterns and generating synthetic scenarios

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

class ClimateEncoder(nn.Module):
    """
    Encoder network for climate VAE
    Maps climate fields to latent distribution parameters
    """
    
    def __init__(self, input_channels=5, latent_dim=32, hidden_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Convolutional encoder for spatial climate data
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 4, stride=2, padding=1),  # H/2 x W/2
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            nn.Conv2d(hidden_dim, hidden_dim*2, 4, stride=2, padding=1),    # H/4 x W/4
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 4, stride=2, padding=1),  # H/8 x W/8
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            
            nn.Conv2d(hidden_dim*4, hidden_dim*8, 4, stride=2, padding=1),  # H/16 x W/16
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),
        )
        
        # Calculate the size after convolutions (assuming 128x128 input -> 8x8 output)
        self.conv_output_size = hidden_dim * 8 * 8 * 8  # This will be calculated dynamically
        
        # Linear layers for latent parameters
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
    
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input climate field [B, C, H, W]
            
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        # Convolutional encoding
        conv_out = self.conv_layers(x)  # [B, hidden_dim*8, H/16, W/16]
        
        # Flatten for linear layers
        flat = conv_out.view(conv_out.size(0), -1)  # [B, conv_output_size]
        
        # Update conv_output_size if needed (first forward pass)
        if hasattr(self, '_first_forward') and self._first_forward:
            self.conv_output_size = flat.size(1)
            self.fc_mu = nn.Linear(self.conv_output_size, self.latent_dim).to(x.device)
            self.fc_logvar = nn.Linear(self.conv_output_size, self.latent_dim).to(x.device)
            self._first_forward = False
        
        # Latent distribution parameters
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        
        return mu, logvar

class ClimateDecoder(nn.Module):
    """
    Decoder network for climate VAE
    Maps latent codes to climate field reconstructions
    """
    
    def __init__(self, latent_dim=32, output_channels=5, hidden_dim=64, output_size=128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        # Calculate the starting size for transposed convolutions
        self.start_size = output_size // 16  # 8 for 128x128 output
        self.fc_input_size = hidden_dim * 8 * self.start_size * self.start_size
        
        # Linear layer to expand latent code
        self.fc_decode = nn.Linear(latent_dim, self.fc_input_size)
        
        # Transposed convolutional decoder
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, 4, stride=2, padding=1),  # 2x size
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 4, stride=2, padding=1),  # 4x size
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1),    # 8x size
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim, output_channels, 4, stride=2, padding=1), # 16x size = output_size
            nn.Tanh()  # Climate anomalies are typically in [-1, 1] range after normalization
        )
    
    def forward(self, z):
        """
        Forward pass through decoder
        
        Args:
            z: Latent code [B, latent_dim]
            
        Returns:
            reconstructed climate field [B, output_channels, output_size, output_size]
        """
        # Expand latent code
        h = F.relu(self.fc_decode(z))  # [B, fc_input_size]
        
        # Reshape for deconvolution
        h = h.view(-1, self.hidden_dim*8, self.start_size, self.start_size)
        
        # Deconvolutional decoding
        reconstruction = self.deconv_layers(h)
        
        return reconstruction

class ClimateVAE(nn.Module):
    """
    Variational AutoEncoder for climate data
    
    Key concepts from syllabus:
    - Variational inference
    - Evidence Lower Bound (ELBO)
    - Reparameterization trick
    - Latent variable models
    - Generative modeling
    """
    
    def __init__(self, input_channels=5, latent_dim=32, hidden_dim=64, input_size=128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Encoder and decoder networks
        self.encoder = ClimateEncoder(input_channels, latent_dim, hidden_dim)
        self.decoder = ClimateDecoder(latent_dim, input_channels, hidden_dim, input_size)
        
        # Set flag for dynamic sizing
        self.encoder._first_forward = True
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for differentiable sampling
        z = μ + σ ⊙ ε, where ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent code to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Returns:
            reconstruction: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, n_samples, device='cpu'):
        """
        Generate samples from the learned climate distribution
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated climate fields [n_samples, channels, H, W]
        """
        with torch.no_grad():
            # Sample from prior distribution N(0, I)
            z = torch.randn(n_samples, self.latent_dim).to(device)
            
            # Decode to climate fields
            samples = self.decode(z)
            
        return samples
    
    def interpolate(self, x1, x2, n_steps=10):
        """
        Interpolate between two climate fields in latent space
        
        Args:
            x1, x2: Input climate fields to interpolate between
            n_steps: Number of interpolation steps
            
        Returns:
            Interpolated climate fields
        """
        with torch.no_grad():
            # Encode inputs
            mu1, _ = self.encode(x1.unsqueeze(0))
            mu2, _ = self.encode(x2.unsqueeze(0))
            
            # Create interpolation weights
            alphas = torch.linspace(0, 1, n_steps).unsqueeze(1).to(x1.device)
            
            # Interpolate in latent space
            z_interp = alphas * mu2 + (1 - alphas) * mu1
            
            # Decode interpolated latent codes
            interpolations = self.decode(z_interp)
            
        return interpolations

def vae_loss_function(reconstruction, x, mu, logvar, beta=1.0):
    """
    VAE loss function: ELBO = -Reconstruction Loss - β * KL Divergence
    
    Args:
        reconstruction: Reconstructed input
        x: Original input
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term (β-VAE)
    """
    # Reconstruction loss (MSE for climate data)
    reconstruction_loss = F.mse_loss(reconstruction, x, reduction='sum')
    
    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss

class ClimateVAEPredictor(nn.Module):
    """
    Climate prediction model using VAE latent representations
    Combines VAE with your existing prediction models
    """
    
    def __init__(self, vae_model, predictor_model, freeze_vae=True):
        super().__init__()
        
        self.vae = vae_model
        self.predictor = predictor_model
        
        if freeze_vae:
            # Freeze VAE parameters during prediction training
            for param in self.vae.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass: encode to latent space, predict, decode
        
        Args:
            x: Input climate sequence [B, T, C, H, W]
            
        Returns:
            predictions: Predicted climate fields
            latent_representations: Latent codes for interpretability
        """
        B, T, C, H, W = x.shape
        
        # Encode each timestep to latent space
        latent_codes = []
        for t in range(T):
            mu, logvar = self.vae.encode(x[:, t])
            z = self.vae.reparameterize(mu, logvar)
            latent_codes.append(z)
        
        latent_sequence = torch.stack(latent_codes, dim=1)  # [B, T, latent_dim]
        
        # Predict next latent code
        if hasattr(self.predictor, 'predict_latent'):
            # Custom latent predictor
            next_latent = self.predictor.predict_latent(latent_sequence)
        else:
            # Use existing predictor (may need adaptation)
            # For simplicity, predict in original space and encode
            original_pred, _ = self.predictor(x)
            next_latent_mu, _ = self.vae.encode(original_pred)
            next_latent = next_latent_mu
        
        # Decode predicted latent code
        prediction = self.vae.decode(next_latent)
        
        return prediction, latent_sequence

def train_climate_vae(vae, train_loader, val_loader, epochs=100, lr=1e-3, beta=1.0):
    """Train the Climate VAE"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print(f"🧠 Training Climate VAE on {device}")
    
    for epoch in range(epochs):
        # Training phase
        vae.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, logvar = vae(data)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss_function(
                reconstruction, data, mu, logvar, beta=beta
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
        
        # Validation phase
        vae.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                
                reconstruction, mu, logvar = vae(data)
                total_loss, recon_loss, kl_loss = vae_loss_function(
                    reconstruction, data, mu, logvar, beta=beta
                )
                
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        # Average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'           Train Recon: {train_recon_loss/len(train_loader.dataset):.4f}, '
                  f'Train KL: {train_kl_loss/len(train_loader.dataset):.4f}')
    
    return train_losses, val_losses

def demonstrate_climate_vae():
    """Demonstration of Climate VAE"""
    print("🧠 Demonstrating Climate Variational AutoEncoder")
    
    # Generate synthetic climate data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic climate patterns
    n_samples = 200
    channels = 5
    size = 64  # Smaller for demo
    
    # Generate climate fields with different patterns
    climate_data = []
    
    for i in range(n_samples):
        # Create different climate patterns
        if i < n_samples // 3:
            # Pattern 1: Circular temperature anomaly
            x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            pattern = np.exp(-(x**2 + y**2) / 0.3)
        elif i < 2 * n_samples // 3:
            # Pattern 2: Linear gradient
            x = np.linspace(-1, 1, size)
            pattern = np.outer(x, np.ones(size))
        else:
            # Pattern 3: Random field
            pattern = np.random.randn(size, size)
            pattern = np.convolve(pattern.flatten(), np.ones(5)/5, mode='same').reshape(size, size)
        
        # Create multi-channel climate field
        field = np.zeros((channels, size, size))
        for c in range(channels):
            field[c] = pattern + 0.1 * np.random.randn(size, size)
        
        climate_data.append(field)
    
    climate_data = np.array(climate_data)
    climate_tensor = torch.FloatTensor(climate_data)
    
    # Create train/val split
    train_size = int(0.8 * n_samples)
    train_data = climate_tensor[:train_size]
    val_data = climate_tensor[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_data, torch.zeros(len(train_data)))  # Dummy labels
    val_dataset = TensorDataset(val_data, torch.zeros(len(val_data)))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize VAE
    vae = ClimateVAE(input_channels=channels, latent_dim=16, hidden_dim=32, input_size=size)
    
    # Train VAE
    train_losses, val_losses = train_climate_vae(
        vae, train_loader, val_loader, epochs=50, lr=1e-3, beta=1.0
    )
    
    # Generate samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    
    # Sample from learned distribution
    generated_samples = vae.sample(9, device=device)
    generated_samples = generated_samples.cpu()
    
    # Visualize results
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('VAE Training Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Original samples
    for i in range(3):
        sample_idx = i * 20
        axes[1, i].imshow(climate_tensor[sample_idx, 0], cmap='coolwarm', origin='lower')
        axes[1, i].set_title(f'Original Sample {i+1}')
        axes[1, i].axis('off')
    
    # Reconstructions
    vae.eval()
    with torch.no_grad():
        test_samples = climate_tensor[:3].to(device)
        reconstructions, _, _ = vae(test_samples)
        reconstructions = reconstructions.cpu()
    
    for i in range(3):
        axes[2, i].imshow(reconstructions[i, 0], cmap='coolwarm', origin='lower')
        axes[2, i].set_title(f'Reconstruction {i+1}')
        axes[2, i].axis('off')
    
    # Generated samples
    for i in range(3):
        axes[3, i].imshow(generated_samples[i, 0], cmap='coolwarm', origin='lower')
        axes[3, i].set_title(f'Generated Sample {i+1}')
        axes[3, i].axis('off')
    
    # Remove empty subplot
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/climate_vae_demonstration.png', dpi=150, bbox_inches='tight')
    print(f"✅ Climate VAE demonstration saved to figures/climate_vae_demonstration.png")
    
    # Analyze latent space
    print("\n📊 Latent Space Analysis:")
    
    with torch.no_grad():
        all_data = climate_tensor.to(device)
        mu, logvar = vae.encode(all_data)
        latent_codes = mu.cpu().numpy()
    
    # Create labels for different patterns
    labels = ['Circular'] * (n_samples // 3) + ['Linear'] * (n_samples // 3) + ['Random'] * (n_samples - 2 * (n_samples // 3))
    
    # t-SNE visualization of latent space
    if latent_codes.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_codes)
    else:
        latent_2d = latent_codes
    
    plt.figure(figsize=(10, 8))
    for pattern_type in ['Circular', 'Linear', 'Random']:
        mask = np.array(labels) == pattern_type
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   label=pattern_type, alpha=0.6, s=50)
    
    plt.title('Climate VAE Latent Space (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/climate_vae_latent_space.png', dpi=150, bbox_inches='tight')
    print(f"✅ Latent space visualization saved to figures/climate_vae_latent_space.png")
    
    return vae, climate_tensor, train_losses, val_losses

if __name__ == "__main__":
    # Create figures directory if needed
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Run demonstration
    vae, data, train_losses, val_losses = demonstrate_climate_vae()
    
    print("\n🎯 Integration with your existing project:")
    print("1. Use ClimateVAE to learn compact representations of your ERA5 data")
    print("2. Apply ClimateVAEPredictor for prediction in latent space")
    print("3. Generate synthetic climate scenarios for data augmentation")
    print("4. Use latent interpolation to study climate transitions")
    print("5. Combine with other probabilistic models (HMM, MRF) for structured generation")