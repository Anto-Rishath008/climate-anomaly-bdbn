# src/climate_hmm.py
# Hidden Markov Model for Climate Regime Detection
# Models temporal transitions between climate states (El Niño, La Niña, Normal)

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns

class ClimateHMM:
    """
    Hidden Markov Model for climate regime detection and prediction.
    
    Key concepts from syllabus:
    - Hidden states: Climate regimes (El Niño, La Niña, Normal)
    - Observations: Temperature anomaly patterns
    - Temporal Markov property: P(S_t | S_{t-1}, ...) = P(S_t | S_{t-1})
    - Forward-backward algorithm for inference
    - Viterbi algorithm for most likely state sequence
    """
    
    def __init__(self, n_states=3, n_features=None):
        """
        Args:
            n_states: Number of hidden climate states (default: 3 for El Niño/La Niña/Normal)
            n_features: Dimensionality of observations (climate features)
        """
        self.n_states = n_states
        self.n_features = n_features
        
        # HMM parameters
        self.initial_probs = None      # π: Initial state probabilities
        self.transition_matrix = None  # A: State transition probabilities
        self.emission_params = None    # B: Emission (observation) parameters
        
        # State names for interpretability
        self.state_names = ['Normal', 'El Niño', 'La Niña'] if n_states == 3 else [f'State_{i}' for i in range(n_states)]
    
    def initialize_parameters(self, observations):
        """Initialize HMM parameters using observations"""
        T, self.n_features = observations.shape
        
        # Initialize with uniform/random distributions
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        # Random transition matrix (row-stochastic)
        self.transition_matrix = np.random.rand(self.n_states, self.n_states)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize emission parameters using K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        cluster_labels = kmeans.fit_predict(observations)
        
        self.emission_params = []
        for state in range(self.n_states):
            state_obs = observations[cluster_labels == state]
            if len(state_obs) > 0:
                mean = np.mean(state_obs, axis=0)
                cov = np.cov(state_obs.T) + 1e-6 * np.eye(self.n_features)  # Add regularization
            else:
                mean = np.random.randn(self.n_features)
                cov = np.eye(self.n_features)
            
            self.emission_params.append({'mean': mean, 'cov': cov})
    
    def emission_probability(self, observation, state):
        """
        Compute P(observation | state)
        Using multivariate Gaussian distributions
        """
        mean = self.emission_params[state]['mean']
        cov = self.emission_params[state]['cov']
        
        try:
            prob = multivariate_normal.pdf(observation, mean=mean, cov=cov)
            return max(prob, 1e-100)  # Avoid numerical issues
        except:
            return 1e-100
    
    def forward_algorithm(self, observations):
        """
        Forward algorithm for computing forward probabilities
        α_t(i) = P(o_1, ..., o_t, q_t = i | λ)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization (t=0)
        for state in range(self.n_states):
            alpha[0, state] = self.initial_probs[state] * self.emission_probability(observations[0], state)
        
        # Recursion (t=1 to T-1)
        for t in range(1, T):
            for state in range(self.n_states):
                alpha[t, state] = sum(
                    alpha[t-1, prev_state] * self.transition_matrix[prev_state, state] * 
                    self.emission_probability(observations[t], state)
                    for prev_state in range(self.n_states)
                )
        
        return alpha
    
    def backward_algorithm(self, observations):
        """
        Backward algorithm for computing backward probabilities
        β_t(i) = P(o_{t+1}, ..., o_T | q_t = i, λ)
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization (t=T-1)
        beta[T-1, :] = 1.0
        
        # Recursion (t=T-2 to 0)
        for t in range(T-2, -1, -1):
            for state in range(self.n_states):
                beta[t, state] = sum(
                    self.transition_matrix[state, next_state] * 
                    self.emission_probability(observations[t+1], next_state) * 
                    beta[t+1, next_state]
                    for next_state in range(self.n_states)
                )
        
        return beta
    
    def viterbi_algorithm(self, observations):
        """
        Viterbi algorithm for finding most likely state sequence
        Concepts: Dynamic programming, maximum likelihood path
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        for state in range(self.n_states):
            delta[0, state] = self.initial_probs[state] * self.emission_probability(observations[0], state)
            psi[0, state] = 0
        
        # Recursion
        for t in range(1, T):
            for state in range(self.n_states):
                probs = [delta[t-1, prev_state] * self.transition_matrix[prev_state, state] 
                        for prev_state in range(self.n_states)]
                delta[t, state] = max(probs) * self.emission_probability(observations[t], state)
                psi[t, state] = np.argmax(probs)
        
        # Termination
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1, :])
        
        # Path backtracking
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path, np.max(delta[T-1, :])
    
    def baum_welch_step(self, observations):
        """
        One step of Baum-Welch (EM) algorithm for parameter learning
        """
        T = len(observations)
        
        # E-step: Compute forward and backward probabilities
        alpha = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations)
        
        # Compute gamma (state probabilities) and xi (transition probabilities)
        gamma = np.zeros((T, self.n_states))
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T):
            normalizer = np.sum(alpha[t, :] * beta[t, :])
            if normalizer > 0:
                gamma[t, :] = (alpha[t, :] * beta[t, :]) / normalizer
        
        for t in range(T-1):
            normalizer = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.transition_matrix[i, j] * 
                                  self.emission_probability(observations[t+1], j) * beta[t+1, j])
                    normalizer += xi[t, i, j]
            
            if normalizer > 0:
                xi[t, :, :] /= normalizer
        
        # M-step: Update parameters
        # Update initial probabilities
        self.initial_probs = gamma[0, :]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                if denominator > 0:
                    self.transition_matrix[i, j] = numerator / denominator
        
        # Update emission parameters
        for state in range(self.n_states):
            weights = gamma[:, state]
            if np.sum(weights) > 0:
                # Weighted mean
                weighted_mean = np.sum(weights[:, np.newaxis] * observations, axis=0) / np.sum(weights)
                
                # Weighted covariance
                diff = observations - weighted_mean
                weighted_cov = np.sum(weights[:, np.newaxis, np.newaxis] * 
                                    diff[:, :, np.newaxis] * diff[:, np.newaxis, :], axis=0) / np.sum(weights)
                weighted_cov += 1e-6 * np.eye(self.n_features)  # Regularization
                
                self.emission_params[state]['mean'] = weighted_mean
                self.emission_params[state]['cov'] = weighted_cov
        
        return gamma, xi
    
    def fit(self, observations, max_iterations=100, tolerance=1e-6):
        """
        Fit HMM parameters using Baum-Welch algorithm
        """
        self.initialize_parameters(observations)
        
        prev_log_likelihood = float('-inf')
        
        for iteration in range(max_iterations):
            # E-step and M-step
            gamma, xi = self.baum_welch_step(observations)
            
            # Compute log-likelihood
            alpha = self.forward_algorithm(observations)
            log_likelihood = np.log(np.sum(alpha[-1, :]) + 1e-100)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = log_likelihood
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Log-likelihood: {log_likelihood:.6f}")
        
        return log_likelihood
    
    def predict_next_state(self, current_state):
        """Predict next climate state given current state"""
        return np.random.choice(self.n_states, p=self.transition_matrix[current_state, :])
    
    def predict_sequence(self, initial_state, length):
        """Generate a sequence of climate states"""
        sequence = [initial_state]
        current_state = initial_state
        
        for _ in range(length - 1):
            next_state = self.predict_next_state(current_state)
            sequence.append(next_state)
            current_state = next_state
        
        return sequence
    
    def visualize_model(self, save_path=None):
        """Visualize HMM structure and parameters"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Transition matrix heatmap
        sns.heatmap(self.transition_matrix, annot=True, cmap='Blues', 
                   xticklabels=self.state_names, yticklabels=self.state_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title('State Transition Matrix')
        axes[0, 0].set_xlabel('Next State')
        axes[0, 0].set_ylabel('Current State')
        
        # 2. Initial state probabilities
        axes[0, 1].bar(self.state_names, self.initial_probs)
        axes[0, 1].set_title('Initial State Probabilities')
        axes[0, 1].set_ylabel('Probability')
        
        # 3. Emission means (if 2D)
        if self.n_features == 2:
            for state in range(self.n_states):
                mean = self.emission_params[state]['mean']
                axes[1, 0].scatter(mean[0], mean[1], s=100, label=self.state_names[state])
            axes[1, 0].set_title('Emission Distribution Means')
            axes[1, 0].set_xlabel('Feature 1')
            axes[1, 0].set_ylabel('Feature 2')
            axes[1, 0].legend()
        
        # 4. Model summary
        axes[1, 1].text(0.1, 0.9, f'Number of States: {self.n_states}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f'Feature Dimension: {self.n_features}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, 'State Names:', transform=axes[1, 1].transAxes)
        for i, name in enumerate(self.state_names):
            axes[1, 1].text(0.1, 0.6 - i*0.1, f'  {i}: {name}', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

class ClimateHMMPredictor(nn.Module):
    """
    Neural network that incorporates HMM climate regime information
    Combines regime detection with your existing forecasting models
    """
    
    def __init__(self, base_model, hmm_model, regime_weight=0.2):
        super().__init__()
        self.base_model = base_model
        self.hmm_model = hmm_model
        self.regime_weight = regime_weight
        
        # Regime-specific adjustment layers
        self.regime_adjustments = nn.ModuleList([
            nn.Linear(2, 2) for _ in range(hmm_model.n_states)  # Adjust mean and log_var
        ])
    
    def forward(self, x, regime_probs=None):
        """
        Forward pass with regime information
        Args:
            x: Input climate sequence
            regime_probs: Probabilistic regime assignments (optional)
        """
        # Get base predictions
        mu, log_var = self.base_model(x)
        
        if regime_probs is not None:
            # Adjust predictions based on climate regime
            batch_size = mu.shape[0]
            
            regime_adjustments = torch.zeros_like(torch.cat([mu, log_var], dim=1))
            
            for regime in range(self.hmm_model.n_states):
                regime_adj = self.regime_adjustments[regime](torch.cat([mu, log_var], dim=1))
                regime_weight = regime_probs[:, regime:regime+1, :, :]
                regime_adjustments += regime_weight * regime_adj
            
            # Apply regime-specific adjustments
            adjusted = torch.cat([mu, log_var], dim=1) + self.regime_weight * regime_adjustments
            mu_adj = adjusted[:, :1, :, :]
            log_var_adj = adjusted[:, 1:, :, :]
            
            return mu_adj, log_var_adj
        
        return mu, log_var

def demonstrate_climate_hmm():
    """Demonstration of HMM for climate regime detection"""
    print("🔄 Demonstrating Hidden Markov Model for Climate Regimes")
    
    # Generate synthetic climate data with regime transitions
    np.random.seed(42)
    T = 200  # Time points
    
    # Define true regimes with different characteristics
    true_regimes = {
        0: {'mean': np.array([0.0, 0.0]), 'cov': np.array([[1.0, 0.2], [0.2, 1.0]])},      # Normal
        1: {'mean': np.array([2.0, 1.5]), 'cov': np.array([[1.5, 0.0], [0.0, 1.5]])},      # El Niño  
        2: {'mean': np.array([-1.5, -1.0]), 'cov': np.array([[1.0, -0.3], [-0.3, 1.0]])}   # La Niña
    }
    
    # Generate regime sequence with realistic transitions
    true_transition = np.array([
        [0.85, 0.10, 0.05],  # Normal -> Normal, El Niño, La Niña
        [0.30, 0.60, 0.10],  # El Niño -> Normal, El Niño, La Niña  
        [0.30, 0.05, 0.65]   # La Niña -> Normal, El Niño, La Niña
    ])
    
    # Generate true state sequence
    true_states = [0]  # Start in normal state
    for t in range(1, T):
        next_state = np.random.choice(3, p=true_transition[true_states[-1], :])
        true_states.append(next_state)
    
    # Generate observations based on true states
    observations = []
    for t in range(T):
        state = true_states[t]
        obs = np.random.multivariate_normal(
            true_regimes[state]['mean'], 
            true_regimes[state]['cov']
        )
        observations.append(obs)
    
    observations = np.array(observations)
    
    # Fit HMM
    print("Fitting HMM to climate data...")
    hmm = ClimateHMM(n_states=3, n_features=2)
    log_likelihood = hmm.fit(observations, max_iterations=50)
    
    # Decode most likely state sequence
    predicted_states, _ = hmm.viterbi_algorithm(observations)
    
    # Compute accuracy
    accuracy = np.mean(np.array(true_states) == predicted_states)
    print(f"✅ State sequence accuracy: {accuracy:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot observations colored by true states
    colors = ['blue', 'red', 'green']
    state_names = ['Normal', 'El Niño', 'La Niña']
    
    for state in range(3):
        mask = np.array(true_states) == state
        axes[0, 0].scatter(observations[mask, 0], observations[mask, 1], 
                          c=colors[state], label=f'True {state_names[state]}', alpha=0.6)
    axes[0, 0].set_title('Observations (True States)')
    axes[0, 0].set_xlabel('Temperature Anomaly 1')
    axes[0, 0].set_ylabel('Temperature Anomaly 2')
    axes[0, 0].legend()
    
    # Plot observations colored by predicted states
    for state in range(3):
        mask = predicted_states == state
        axes[0, 1].scatter(observations[mask, 0], observations[mask, 1], 
                          c=colors[state], label=f'Predicted {state_names[state]}', alpha=0.6)
    axes[0, 1].set_title('Observations (Predicted States)')
    axes[0, 1].set_xlabel('Temperature Anomaly 1')
    axes[0, 1].set_ylabel('Temperature Anomaly 2')
    axes[0, 1].legend()
    
    # Plot state sequences over time
    time_points = np.arange(T)
    axes[1, 0].plot(time_points, true_states, 'b-', label='True States', alpha=0.7)
    axes[1, 0].plot(time_points, predicted_states, 'r--', label='Predicted States', alpha=0.7)
    axes[1, 0].set_title('Climate Regime Sequences')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Climate State')
    axes[1, 0].set_yticks([0, 1, 2])
    axes[1, 0].set_yticklabels(state_names)
    axes[1, 0].legend()
    
    # Show transition matrices comparison
    sns.heatmap(hmm.transition_matrix, annot=True, cmap='Blues', 
               xticklabels=state_names, yticklabels=state_names, ax=axes[1, 1])
    axes[1, 1].set_title('Learned Transition Matrix')
    
    plt.tight_layout()
    plt.savefig('figures/hmm_demonstration.png', dpi=150, bbox_inches='tight')
    print(f"✅ HMM demonstration saved to figures/hmm_demonstration.png")
    
    return hmm, true_states, predicted_states, observations

if __name__ == "__main__":
    # Create figures directory if needed
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Run demonstration
    hmm, true_states, pred_states, obs = demonstrate_climate_hmm()
    
    # Save model visualization
    hmm.visualize_model('figures/hmm_model_structure.png')
    
    print("\n🎯 Integration with your existing project:")
    print("1. Use HMM to detect climate regimes in your ERA5 data")
    print("2. Condition your Bayesian U-Net predictions on current regime")
    print("3. Use regime information for better uncertainty quantification")
    print("4. Predict regime transitions for longer-term forecasting")