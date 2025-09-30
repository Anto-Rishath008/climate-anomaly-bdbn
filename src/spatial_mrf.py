# src/spatial_mrf.py
# Markov Random Field for Climate Spatial Dependencies
# Captures spatial correlations between neighboring climate grid points

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class ClimateMRF:
    """
    Markov Random Field for modeling spatial dependencies in climate anomalies.
    
    Key concepts from syllabus:
    - Markov property: P(X_i | X_rest) = P(X_i | X_neighbors)
    - Spatial correlation modeling
    - Energy-based probability distributions
    """
    
    def __init__(self, height, width, neighbor_type='4-connected', beta=1.0):
        """
        Args:
            height, width: Grid dimensions
            neighbor_type: '4-connected' or '8-connected'
            beta: Interaction strength parameter
        """
        self.H, self.W = height, width
        self.N = height * width
        self.beta = beta
        self.neighbor_type = neighbor_type
        
        # Build adjacency matrix for spatial neighbors
        self.adjacency = self._build_adjacency_matrix()
        
    def _build_adjacency_matrix(self):
        """Build sparse adjacency matrix for grid neighbors"""
        row_idx, col_idx = [], []
        
        for i in range(self.H):
            for j in range(self.W):
                node_id = i * self.W + j
                
                # 4-connected neighbors
                neighbors = []
                if i > 0: neighbors.append((i-1, j))        # up
                if i < self.H-1: neighbors.append((i+1, j)) # down
                if j > 0: neighbors.append((i, j-1))        # left
                if j < self.W-1: neighbors.append((i, j+1)) # right
                
                # Add 8-connected diagonal neighbors if specified
                if self.neighbor_type == '8-connected':
                    if i > 0 and j > 0: neighbors.append((i-1, j-1))
                    if i > 0 and j < self.W-1: neighbors.append((i-1, j+1))
                    if i < self.H-1 and j > 0: neighbors.append((i+1, j-1))
                    if i < self.H-1 and j < self.W-1: neighbors.append((i+1, j+1))
                
                for ni, nj in neighbors:
                    neighbor_id = ni * self.W + nj
                    row_idx.append(node_id)
                    col_idx.append(neighbor_id)
        
        # Create sparse adjacency matrix
        data = np.ones(len(row_idx))
        adjacency = csr_matrix((data, (row_idx, col_idx)), shape=(self.N, self.N))
        return adjacency
    
    def energy_function(self, x):
        """
        Compute MRF energy: E(x) = -β * Σ x_i * x_j (for neighboring i,j)
        Lower energy = higher probability
        """
        x_flat = x.flatten()
        # Pairwise interaction energy
        pairwise_energy = -self.beta * (x_flat @ self.adjacency @ x_flat) / 2
        return pairwise_energy
    
    def probability_unnormalized(self, x):
        """Unnormalized probability: exp(-E(x))"""
        energy = self.energy_function(x)
        return np.exp(-energy)
    
    def conditional_probability(self, x, i, j, value):
        """
        P(x_ij = value | x_neighbors)
        Key concept: Markov property - only depends on neighbors
        """
        # Get neighbors of position (i,j)
        node_id = i * self.W + j
        neighbor_ids = self.adjacency[node_id].nonzero()[1]
        
        x_copy = x.copy()
        x_copy[i, j] = value
        
        # Local energy change
        neighbor_sum = sum(x.flatten()[nid] for nid in neighbor_ids)
        local_energy = -self.beta * value * neighbor_sum
        
        return np.exp(-local_energy)
    
    def gibbs_sampling(self, n_samples=1000, burn_in=200, initial_state=None):
        """
        Gibbs sampling for MRF inference
        Concept from syllabus: MCMC sampling
        """
        if initial_state is None:
            x = np.random.randn(self.H, self.W)
        else:
            x = initial_state.copy()
        
        samples = []
        
        for sample_idx in range(n_samples + burn_in):
            # Sample each pixel given its neighbors
            for i in range(self.H):
                for j in range(self.W):
                    # Compute conditional probabilities for different values
                    node_id = i * self.W + j
                    neighbor_ids = self.adjacency[node_id].nonzero()[1]
                    neighbor_sum = sum(x.flatten()[nid] for nid in neighbor_ids)
                    
                    # Sample from Gaussian conditional (for continuous MRF)
                    conditional_mean = self.beta * neighbor_sum / len(neighbor_ids) if len(neighbor_ids) > 0 else 0
                    x[i, j] = np.random.normal(conditional_mean, 1.0)
            
            # Store sample after burn-in
            if sample_idx >= burn_in:
                samples.append(x.copy())
        
        return np.array(samples)
    
    def belief_propagation_step(self, messages, beliefs):
        """
        One step of belief propagation for MRF inference
        Concept from syllabus: Message Passing
        """
        new_messages = messages.copy()
        
        for node in range(self.N):
            neighbors = self.adjacency[node].nonzero()[1]
            
            for neighbor in neighbors:
                # Compute message from node to neighbor
                # msg = Σ_x φ(x) * ψ(x, neighbor) * Π(incoming messages except from neighbor)
                incoming_msgs = [messages[n][node] for n in neighbors if n != neighbor]
                
                # Simplified message update (for Gaussian MRF)
                if len(incoming_msgs) > 0:
                    new_messages[node][neighbor] = np.mean(incoming_msgs)
                else:
                    new_messages[node][neighbor] = 0.0
        
        return new_messages

class ClimateMRFPredictor(nn.Module):
    """
    Neural network that incorporates MRF spatial priors
    Combines your existing Bayesian U-Net with MRF regularization
    """
    
    def __init__(self, base_model, mrf_weight=0.1):
        """
        Args:
            base_model: Your existing BayesianUNet
            mrf_weight: Weight for MRF spatial regularization term
        """
        super().__init__()
        self.base_model = base_model
        self.mrf_weight = mrf_weight
        
    def forward(self, x):
        # Get base predictions
        mu, log_var = self.base_model(x)
        return mu, log_var
    
    def mrf_spatial_loss(self, predictions):
        """
        Add MRF spatial smoothness regularization
        Encourages neighboring pixels to have similar predictions
        """
        B, C, H, W = predictions.shape
        
        # Compute spatial gradients (approximation of MRF energy)
        grad_h = torch.abs(predictions[:, :, 1:, :] - predictions[:, :, :-1, :])
        grad_w = torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1])
        
        # Total variation regularization (MRF-inspired)
        spatial_loss = grad_h.mean() + grad_w.mean()
        return spatial_loss
    
    def compute_loss(self, predictions, targets, prediction_var):
        """
        Enhanced loss with MRF spatial regularization
        """
        # Base NLL loss (from your existing model)
        nll_loss = 0.5 * ((predictions - targets)**2 / prediction_var + torch.log(prediction_var))
        nll_loss = nll_loss.mean()
        
        # MRF spatial regularization
        spatial_loss = self.mrf_spatial_loss(predictions)
        
        # Combined loss
        total_loss = nll_loss + self.mrf_weight * spatial_loss
        return total_loss, nll_loss, spatial_loss

def demonstrate_mrf_climate():
    """Demonstration of MRF for climate data"""
    print("🌐 Demonstrating Markov Random Field for Climate Modeling")
    
    # Create a simple climate field with spatial correlation
    H, W = 32, 32
    mrf = ClimateMRF(H, W, neighbor_type='4-connected', beta=0.5)
    
    # Generate spatially correlated climate field using Gibbs sampling
    print("Generating spatially correlated climate field...")
    samples = mrf.gibbs_sampling(n_samples=100, burn_in=50)
    
    # Take the last sample as our generated field
    climate_field = samples[-1]
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original field
    im1 = axes[0].imshow(climate_field, cmap='coolwarm', origin='lower')
    axes[0].set_title('MRF-Generated Climate Field')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Show spatial correlation
    # Compare with random field
    random_field = np.random.randn(H, W)
    im2 = axes[1].imshow(random_field, cmap='coolwarm', origin='lower')
    axes[1].set_title('Random Field (No Spatial Correlation)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Compute and show energy
    mrf_energy = mrf.energy_function(climate_field)
    random_energy = mrf.energy_function(random_field)
    
    axes[2].bar(['MRF Field', 'Random Field'], [mrf_energy, random_energy])
    axes[2].set_title('Energy Comparison\n(Lower = More Spatially Correlated)')
    axes[2].set_ylabel('Energy')
    
    plt.tight_layout()
    plt.savefig('figures/mrf_demonstration.png', dpi=150, bbox_inches='tight')
    print(f"✅ MRF demonstration saved to figures/mrf_demonstration.png")
    print(f"MRF Energy: {mrf_energy:.3f}, Random Energy: {random_energy:.3f}")
    
    return mrf, climate_field

if __name__ == "__main__":
    # Create figures directory if needed
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Run demonstration
    mrf, field = demonstrate_mrf_climate()
    
    print("\n🎯 Next steps to integrate with your existing project:")
    print("1. Use ClimateMRFPredictor as wrapper around your BayesianUNet")
    print("2. Add MRF spatial loss to your training loop")
    print("3. Use MRF for spatial smoothing of predictions")
    print("4. Implement belief propagation for efficient inference")