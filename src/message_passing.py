# src/message_passing.py
# Message Passing Algorithms for Climate Inference
# Implements Belief Propagation and Loopy BP for spatial-temporal climate models

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx

class Factor:
    """
    Represents a factor (potential) in a factor graph
    Used for modeling local dependencies in climate data
    """
    
    def __init__(self, variables, potential_func):
        """
        Args:
            variables: List of variable indices connected to this factor
            potential_func: Function that computes potential values
        """
        self.variables = variables
        self.potential_func = potential_func
        self.dimension = len(variables)
    
    def compute_potential(self, assignment):
        """Compute potential value for a given variable assignment"""
        return self.potential_func(assignment)

class FactorGraph:
    """
    Factor graph representation for climate spatial-temporal modeling
    
    Concepts from syllabus:
    - Factor graphs: Bipartite graphs with variable and factor nodes
    - Message passing: Communication between variables and factors
    - Belief propagation: Exact inference on tree-structured graphs
    - Loopy BP: Approximate inference on graphs with cycles
    """
    
    def __init__(self):
        self.variables = {}        # Variable nodes
        self.factors = {}          # Factor nodes
        self.var_to_factors = defaultdict(list)  # Variable -> connected factors
        self.factor_to_vars = defaultdict(list)  # Factor -> connected variables
        
        # Message storage: messages[from_node][to_node]
        self.var_to_factor_messages = defaultdict(lambda: defaultdict(dict))
        self.factor_to_var_messages = defaultdict(lambda: defaultdict(dict))
    
    def add_variable(self, var_id, domain_size):
        """Add a variable node (e.g., climate grid point)"""
        self.variables[var_id] = {'domain_size': domain_size}
    
    def add_factor(self, factor_id, variables, potential_func):
        """Add a factor node (e.g., spatial correlation, temporal dependency)"""
        factor = Factor(variables, potential_func)
        self.factors[factor_id] = factor
        
        # Update adjacency lists
        for var in variables:
            self.var_to_factors[var].append(factor_id)
            self.factor_to_vars[factor_id].append(var)
    
    def initialize_messages(self, init_type='uniform'):
        """Initialize all messages"""
        
        # Variable to factor messages
        for var_id in self.variables:
            domain_size = self.variables[var_id]['domain_size']
            
            if init_type == 'uniform':
                init_msg = np.ones(domain_size) / domain_size
            elif init_type == 'random':
                init_msg = np.random.rand(domain_size)
                init_msg = init_msg / np.sum(init_msg)
            
            for factor_id in self.var_to_factors[var_id]:
                self.var_to_factor_messages[var_id][factor_id] = init_msg.copy()
        
        # Factor to variable messages
        for factor_id in self.factors:
            for var_id in self.factor_to_vars[factor_id]:
                domain_size = self.variables[var_id]['domain_size']
                if init_type == 'uniform':
                    init_msg = np.ones(domain_size) / domain_size
                elif init_type == 'random':
                    init_msg = np.random.rand(domain_size)
                    init_msg = init_msg / np.sum(init_msg)
                
                self.factor_to_var_messages[factor_id][var_id] = init_msg.copy()
    
    def variable_to_factor_message(self, var_id, factor_id):
        """
        Compute message from variable to factor
        m_{X_i → f}(x_i) = ∏_{g ∈ N(X_i) \\ f} m_{g → X_i}(x_i)
        """
        domain_size = self.variables[var_id]['domain_size']
        message = np.ones(domain_size)
        
        # Multiply incoming messages from all factors except the target
        for other_factor_id in self.var_to_factors[var_id]:
            if other_factor_id != factor_id:
                incoming_msg = self.factor_to_var_messages[other_factor_id][var_id]
                message = message * incoming_msg
        
        # Normalize
        normalizer = np.sum(message)
        if normalizer > 0:
            message = message / normalizer
        else:
            message = np.ones(domain_size) / domain_size
        
        return message
    
    def factor_to_variable_message(self, factor_id, var_id):
        """
        Compute message from factor to variable
        m_{f → X_i}(x_i) = ∑_{x_j: j ∈ N(f) \\ i} f(x_N(f)) ∏_{j ∈ N(f) \\ i} m_{X_j → f}(x_j)
        """
        factor = self.factors[factor_id]
        variables = self.factor_to_vars[factor_id]
        target_var_idx = variables.index(var_id)
        other_vars = [v for v in variables if v != var_id]
        
        domain_size = self.variables[var_id]['domain_size']
        message = np.zeros(domain_size)
        
        # If factor involves only one variable, return uniform
        if len(variables) == 1:
            return np.ones(domain_size) / domain_size
        
        # For simplicity, assume binary variables and pairwise factors
        if len(variables) == 2:
            other_var = other_vars[0]
            other_domain_size = self.variables[other_var]['domain_size']
            
            for target_val in range(domain_size):
                marginal = 0
                for other_val in range(other_domain_size):
                    # Create assignment
                    if target_var_idx == 0:
                        assignment = [target_val, other_val]
                    else:
                        assignment = [other_val, target_val]
                    
                    # Get potential and incoming message
                    potential = factor.compute_potential(assignment)
                    incoming_msg = self.var_to_factor_messages[other_var][factor_id][other_val]
                    
                    marginal += potential * incoming_msg
                
                message[target_val] = marginal
        
        # Normalize
        normalizer = np.sum(message)
        if normalizer > 0:
            message = message / normalizer
        else:
            message = np.ones(domain_size) / domain_size
        
        return message
    
    def belief_propagation_step(self):
        """One step of belief propagation"""
        
        # Update variable to factor messages
        for var_id in self.variables:
            for factor_id in self.var_to_factors[var_id]:
                new_msg = self.variable_to_factor_message(var_id, factor_id)
                self.var_to_factor_messages[var_id][factor_id] = new_msg
        
        # Update factor to variable messages  
        for factor_id in self.factors:
            for var_id in self.factor_to_vars[factor_id]:
                new_msg = self.factor_to_variable_message(factor_id, var_id)
                self.factor_to_var_messages[factor_id][var_id] = new_msg
    
    def compute_marginals(self):
        """
        Compute marginal beliefs for each variable
        b_i(x_i) = ∏_{f ∈ N(X_i)} m_{f → X_i}(x_i)
        """
        marginals = {}
        
        for var_id in self.variables:
            domain_size = self.variables[var_id]['domain_size']
            belief = np.ones(domain_size)
            
            # Multiply all incoming factor messages
            for factor_id in self.var_to_factors[var_id]:
                incoming_msg = self.factor_to_var_messages[factor_id][var_id]
                belief = belief * incoming_msg
            
            # Normalize
            normalizer = np.sum(belief)
            if normalizer > 0:
                belief = belief / normalizer
            
            marginals[var_id] = belief
        
        return marginals
    
    def loopy_belief_propagation(self, max_iterations=50, tolerance=1e-6):
        """
        Loopy belief propagation for approximate inference
        Returns marginal beliefs and convergence info
        """
        print("Running Loopy Belief Propagation...")
        
        self.initialize_messages()
        
        for iteration in range(max_iterations):
            # Store old messages for convergence check
            old_var_messages = {}
            for var_id in self.variables:
                old_var_messages[var_id] = {}
                for factor_id in self.var_to_factors[var_id]:
                    old_var_messages[var_id][factor_id] = self.var_to_factor_messages[var_id][factor_id].copy()
            
            # One BP step
            self.belief_propagation_step()
            
            # Check convergence
            max_change = 0
            for var_id in self.variables:
                for factor_id in self.var_to_factors[var_id]:
                    old_msg = old_var_messages[var_id][factor_id]
                    new_msg = self.var_to_factor_messages[var_id][factor_id]
                    change = np.max(np.abs(new_msg - old_msg))
                    max_change = max(max_change, change)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Max message change: {max_change:.6f}")
            
            if max_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Compute final marginals
        marginals = self.compute_marginals()
        return marginals, iteration + 1

class ClimateSpatialFactorGraph:
    """
    Specialized factor graph for climate spatial modeling
    Models spatial correlations using pairwise factors
    """
    
    def __init__(self, height, width, n_states=10):
        """
        Args:
            height, width: Climate grid dimensions
            n_states: Number of discrete states per climate variable
        """
        self.H, self.W = height, width
        self.n_states = n_states
        self.factor_graph = FactorGraph()
        
        self._build_spatial_factor_graph()
    
    def _build_spatial_factor_graph(self):
        """Build factor graph for spatial climate correlations"""
        
        # Add variable nodes (one per grid cell)
        for i in range(self.H):
            for j in range(self.W):
                var_id = i * self.W + j
                self.factor_graph.add_variable(var_id, self.n_states)
        
        # Add pairwise spatial factors
        factor_id = 0
        for i in range(self.H):
            for j in range(self.W):
                var_id = i * self.W + j
                
                # Add factors for spatial neighbors
                neighbors = []
                if i < self.H - 1:  # Down neighbor
                    neighbors.append((i+1) * self.W + j)
                if j < self.W - 1:  # Right neighbor
                    neighbors.append(i * self.W + (j+1))
                
                for neighbor_id in neighbors:
                    # Spatial correlation potential (prefer similar values)
                    def spatial_potential(assignment, strength=2.0):
                        val1, val2 = assignment
                        # Higher potential for similar values
                        return np.exp(-strength * abs(val1 - val2) / self.n_states)
                    
                    self.factor_graph.add_factor(
                        factor_id, [var_id, neighbor_id], spatial_potential
                    )
                    factor_id += 1
    
    def infer_spatial_field(self, observations=None, max_iterations=30):
        """
        Infer complete spatial climate field using BP
        Args:
            observations: Dict of {var_id: observed_value} for partially observed field
        """
        
        # If observations provided, clamp observed variables
        if observations:
            for var_id, obs_value in observations.items():
                # Create evidence factor (delta function)
                def evidence_potential(assignment, obs_val=obs_value):
                    return 1.0 if assignment[0] == obs_val else 1e-10
                
                evidence_factor_id = f"evidence_{var_id}"
                self.factor_graph.add_factor(
                    evidence_factor_id, [var_id], evidence_potential
                )
        
        # Run belief propagation
        marginals, iterations = self.factor_graph.loopy_belief_propagation(
            max_iterations=max_iterations
        )
        
        # Convert marginals to 2D field
        inferred_field = np.zeros((self.H, self.W))
        confidence_field = np.zeros((self.H, self.W))
        
        for i in range(self.H):
            for j in range(self.W):
                var_id = i * self.W + j
                belief = marginals[var_id]
                
                # MAP estimate
                inferred_field[i, j] = np.argmax(belief)
                # Confidence (entropy-based)
                entropy = -np.sum(belief * np.log(belief + 1e-10))
                confidence_field[i, j] = 1.0 - entropy / np.log(self.n_states)
        
        return inferred_field, confidence_field, marginals

class MessagePassingClimatePredictor(nn.Module):
    """
    Neural network incorporating message passing for spatial reasoning
    Combines your existing models with structured inference
    """
    
    def __init__(self, base_model, grid_size, n_discrete_states=10):
        super().__init__()
        self.base_model = base_model
        self.grid_size = grid_size
        self.n_states = n_discrete_states
        
        # Discretization parameters
        self.register_buffer('state_centers', 
                           torch.linspace(-3.0, 3.0, n_discrete_states))
        
        # Message passing network for spatial refinement
        self.message_net = nn.Sequential(
            nn.Conv2d(n_discrete_states, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, n_discrete_states, 3, padding=1),
            nn.Softmax(dim=1)
        )
    
    def discretize_predictions(self, continuous_pred):
        """Convert continuous predictions to discrete distributions"""
        B, C, H, W = continuous_pred.shape
        
        # Compute soft assignments to discrete states
        distances = torch.abs(continuous_pred.unsqueeze(2) - 
                             self.state_centers.view(1, 1, -1, 1, 1))
        
        # Soft assignment using Gaussian kernels
        sigma = 0.5
        soft_assignments = torch.exp(-distances**2 / (2 * sigma**2))
        soft_assignments = soft_assignments / (soft_assignments.sum(dim=2, keepdim=True) + 1e-8)
        
        return soft_assignments.squeeze(1)  # [B, n_states, H, W]
    
    def continuous_from_discrete(self, discrete_dist):
        """Convert discrete distribution back to continuous prediction"""
        # Weighted sum using state centers
        continuous = torch.sum(discrete_dist * self.state_centers.view(1, -1, 1, 1), dim=1, keepdim=True)
        return continuous
    
    def forward(self, x, use_message_passing=True):
        """Forward pass with optional message passing refinement"""
        
        # Base prediction
        mu, log_var = self.base_model(x)
        
        if use_message_passing:
            # Discretize predictions
            discrete_mu = self.discretize_predictions(mu)
            
            # Apply message passing network (simulates BP)
            refined_discrete = self.message_net(discrete_mu)
            
            # Convert back to continuous
            refined_mu = self.continuous_from_discrete(refined_discrete)
            
            return refined_mu, log_var
        
        return mu, log_var

def demonstrate_message_passing():
    """Demonstration of message passing for climate inference"""
    print("📡 Demonstrating Message Passing for Climate Inference")
    
    # Create small climate spatial field
    H, W = 8, 8
    n_states = 5
    
    # Build spatial factor graph
    climate_fg = ClimateSpatialFactorGraph(H, W, n_states)
    
    # Create partial observations (simulate missing data scenario)
    np.random.seed(42)
    true_field = np.random.randint(0, n_states, (H, W))
    
    # Smooth the true field to make it more realistic
    from scipy.ndimage import gaussian_filter
    true_field_smooth = gaussian_filter(true_field.astype(float), sigma=1.0)
    true_field = np.round(true_field_smooth).astype(int)
    true_field = np.clip(true_field, 0, n_states - 1)
    
    # Create partial observations (observe only 30% of grid)
    n_observed = int(0.3 * H * W)
    observed_indices = np.random.choice(H * W, n_observed, replace=False)
    
    observations = {}
    for idx in observed_indices:
        i, j = idx // W, idx % W
        var_id = i * W + j
        observations[var_id] = true_field[i, j]
    
    print(f"Observing {len(observations)} out of {H * W} grid points...")
    
    # Run inference
    inferred_field, confidence, marginals = climate_fg.infer_spatial_field(
        observations, max_iterations=50
    )
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # True field
    im1 = axes[0, 0].imshow(true_field, cmap='coolwarm', origin='lower', vmin=0, vmax=n_states-1)
    axes[0, 0].set_title('True Climate Field')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Observed points
    observed_field = np.full((H, W), -1)
    for var_id, value in observations.items():
        i, j = var_id // W, var_id % W
        observed_field[i, j] = value
    
    masked_obs = np.ma.masked_where(observed_field == -1, observed_field)
    im2 = axes[0, 1].imshow(masked_obs, cmap='coolwarm', origin='lower', vmin=0, vmax=n_states-1)
    axes[0, 1].set_title('Partial Observations')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Inferred field
    im3 = axes[0, 2].imshow(inferred_field, cmap='coolwarm', origin='lower', vmin=0, vmax=n_states-1)
    axes[0, 2].set_title('Inferred Field (Message Passing)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Confidence
    im4 = axes[1, 0].imshow(confidence, cmap='viridis', origin='lower', vmin=0, vmax=1)
    axes[1, 0].set_title('Inference Confidence')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Error map
    error_field = np.abs(true_field - inferred_field)
    im5 = axes[1, 1].imshow(error_field, cmap='Reds', origin='lower')
    axes[1, 1].set_title('Absolute Error')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Accuracy statistics
    overall_accuracy = np.mean(true_field == inferred_field)
    unobserved_mask = np.ones((H, W), dtype=bool)
    for var_id in observations.keys():
        i, j = var_id // W, var_id % W
        unobserved_mask[i, j] = False
    
    unobserved_accuracy = np.mean(true_field[unobserved_mask] == inferred_field[unobserved_mask])
    
    axes[1, 2].text(0.1, 0.8, f'Overall Accuracy: {overall_accuracy:.3f}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f'Unobserved Accuracy: {unobserved_accuracy:.3f}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f'Observed Points: {len(observations)}/{H*W}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'Grid Size: {H}×{W}', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f'Discrete States: {n_states}', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Inference Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/message_passing_demonstration.png', dpi=150, bbox_inches='tight')
    print(f"✅ Message passing demonstration saved to figures/message_passing_demonstration.png")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"Unobserved accuracy: {unobserved_accuracy:.3f}")
    
    return climate_fg, true_field, inferred_field

if __name__ == "__main__":
    # Create figures directory if needed
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Run demonstration
    fg, true_field, inferred_field = demonstrate_message_passing()
    
    print("\n🎯 Integration with your existing project:")
    print("1. Use MessagePassingClimatePredictor to refine spatial predictions")
    print("2. Apply BP for handling missing climate data")
    print("3. Use factor graphs for modeling complex spatial-temporal dependencies")
    print("4. Implement approximate inference for large climate grids")