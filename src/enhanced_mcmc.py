# src/enhanced_mcmc.py
# Enhanced MCMC Methods for Climate Uncertainty Quantification
# Implements advanced Markov Chain Monte Carlo sampling techniques

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class MCMCClimateSampler:
    """
    Advanced MCMC sampler for climate model uncertainty quantification
    
    Concepts from syllabus:
    - Markov Chain Monte Carlo
    - Metropolis-Hastings algorithm
    - Gibbs sampling
    - Hamiltonian Monte Carlo (HMC)
    - Ensemble sampling
    """
    
    def __init__(self, model, likelihood_func, prior_func):
        """
        Args:
            model: Climate prediction model (e.g., your BayesianUNet)
            likelihood_func: Data likelihood function P(data | parameters)
            prior_func: Prior distribution P(parameters)
        """
        self.model = model
        self.likelihood_func = likelihood_func
        self.prior_func = prior_func
        
        # Get model parameters
        self.param_shapes = {}
        self.param_sizes = {}
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_shapes[name] = param.shape
                self.param_sizes[name] = param.numel()
                total_params += param.numel()
        
        self.total_params = total_params
        print(f"Total trainable parameters: {total_params:,}")
    
    def flatten_parameters(self):
        """Flatten model parameters into 1D vector"""
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.data.view(-1))
        return torch.cat(params)
    
    def unflatten_parameters(self, flat_params):
        """Restore model parameters from 1D vector"""
        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                size = self.param_sizes[name]
                shape = self.param_shapes[name]
                param.data = flat_params[idx:idx+size].view(shape)
                idx += size
    
    def log_posterior(self, flat_params, X, y):
        """
        Compute log posterior: log P(θ | data) = log P(data | θ) + log P(θ)
        """
        # Update model parameters
        self.unflatten_parameters(flat_params)
        
        # Compute log likelihood
        log_likelihood = self.likelihood_func(self.model, X, y)
        
        # Compute log prior
        log_prior = self.prior_func(flat_params)
        
        return log_likelihood + log_prior
    
    def metropolis_hastings(self, X, y, n_samples=1000, step_size=0.01, burn_in=500):
        """
        Metropolis-Hastings sampling for parameter posterior
        
        Algorithm:
        1. Propose new state: θ' = θ + ε, ε ~ N(0, σ²I)
        2. Compute acceptance ratio: α = min(1, P(θ'|data) / P(θ|data))
        3. Accept with probability α
        """
        print("🎲 Running Metropolis-Hastings sampling...")
        
        # Initialize chain
        current_params = self.flatten_parameters()
        samples = []
        log_probs = []
        accepted = 0
        
        # Initial log probability
        current_log_prob = self.log_posterior(current_params, X, y)
        
        for i in tqdm(range(n_samples + burn_in)):
            # Propose new parameters
            proposal = current_params + step_size * torch.randn_like(current_params)
            
            # Compute log probability of proposal
            proposal_log_prob = self.log_posterior(proposal, X, y)
            
            # Acceptance ratio (in log space)
            log_alpha = proposal_log_prob - current_log_prob
            alpha = min(1.0, torch.exp(log_alpha).item())
            
            # Accept or reject
            if np.random.rand() < alpha:
                current_params = proposal
                current_log_prob = proposal_log_prob
                accepted += 1
            
            # Store sample after burn-in
            if i >= burn_in:
                samples.append(current_params.clone())
                log_probs.append(current_log_prob.item())
        
        acceptance_rate = accepted / (n_samples + burn_in)
        print(f"✅ MH completed. Acceptance rate: {acceptance_rate:.3f}")
        
        return torch.stack(samples), np.array(log_probs), acceptance_rate
    
    def gibbs_sampling(self, X, y, n_samples=1000, burn_in=500):
        """
        Gibbs sampling for structured parameter updates
        Updates parameter groups alternately
        """
        print("🔄 Running Gibbs sampling...")
        
        samples = []
        
        # Group parameters by layer type
        param_groups = self._group_parameters()
        
        for i in tqdm(range(n_samples + burn_in)):
            # Update each parameter group
            for group_name, param_names in param_groups.items():
                self._update_parameter_group(param_names, X, y)
            
            # Store sample after burn-in
            if i >= burn_in:
                samples.append(self.flatten_parameters().clone())
        
        print("✅ Gibbs sampling completed")
        return torch.stack(samples)
    
    def _group_parameters(self):
        """Group parameters by layer type for Gibbs sampling"""
        groups = {
            'conv_weights': [],
            'conv_biases': [],
            'norm_weights': [],
            'norm_biases': [],
            'linear_weights': [],
            'linear_biases': []
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'conv' in name and 'weight' in name:
                    groups['conv_weights'].append(name)
                elif 'conv' in name and 'bias' in name:
                    groups['conv_biases'].append(name)
                elif ('norm' in name or 'bn' in name) and 'weight' in name:
                    groups['norm_weights'].append(name)
                elif ('norm' in name or 'bn' in name) and 'bias' in name:
                    groups['norm_biases'].append(name)
                elif ('linear' in name or 'fc' in name) and 'weight' in name:
                    groups['linear_weights'].append(name)
                elif ('linear' in name or 'fc' in name) and 'bias' in name:
                    groups['linear_biases'].append(name)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups
    
    def _update_parameter_group(self, param_names, X, y, step_size=0.001):
        """Update a group of parameters using Metropolis step"""
        if not param_names:
            return
        
        # Current parameters
        current_params = {}
        for name in param_names:
            current_params[name] = getattr(self.model, name.split('.')[0])[int(name.split('.')[1])] if '.' in name else getattr(self.model, name)
        
        # Current log probability
        current_log_prob = self.log_posterior(self.flatten_parameters(), X, y)
        
        # Propose updates for this group
        for name in param_names:
            param = dict(self.model.named_parameters())[name]
            proposal = param.data + step_size * torch.randn_like(param.data)
            
            # Temporarily update parameter
            old_data = param.data.clone()
            param.data = proposal
            
            # Compute acceptance probability
            proposal_log_prob = self.log_posterior(self.flatten_parameters(), X, y)
            log_alpha = proposal_log_prob - current_log_prob
            alpha = min(1.0, torch.exp(log_alpha).item())
            
            # Accept or reject
            if np.random.rand() >= alpha:
                param.data = old_data  # Reject
            else:
                current_log_prob = proposal_log_prob  # Accept
    
    def hamiltonian_monte_carlo(self, X, y, n_samples=500, step_size=0.001, n_leapfrog=10):
        """
        Hamiltonian Monte Carlo (HMC) sampling
        Uses gradient information for more efficient sampling
        """
        print("⚡ Running Hamiltonian Monte Carlo...")
        
        samples = []
        
        for i in tqdm(range(n_samples)):
            # Current position (parameters)
            q = self.flatten_parameters().clone().requires_grad_(True)
            
            # Sample momentum
            p = torch.randn_like(q)
            
            # Current energy
            current_U = -self.log_posterior(q, X, y)
            current_K = 0.5 * torch.sum(p**2)
            current_H = current_U + current_K
            
            # Leapfrog integration
            q_new = q.clone()
            p_new = p.clone()
            
            # Half step for momentum
            grad_U = torch.autograd.grad(current_U, q, retain_graph=True)[0]
            p_new = p_new - 0.5 * step_size * grad_U
            
            # Leapfrog steps
            for _ in range(n_leapfrog):
                # Full step for position
                q_new = q_new + step_size * p_new
                
                # Compute gradient at new position
                self.unflatten_parameters(q_new)
                new_U = -self.log_posterior(q_new, X, y)
                grad_U = torch.autograd.grad(new_U, q_new, retain_graph=True)[0]
                
                # Full step for momentum (except last step)
                if _ < n_leapfrog - 1:
                    p_new = p_new - step_size * grad_U
            
            # Half step for momentum
            p_new = p_new - 0.5 * step_size * grad_U
            
            # Negate momentum for symmetry
            p_new = -p_new
            
            # Evaluate energies at new state
            new_U = -self.log_posterior(q_new, X, y)
            new_K = 0.5 * torch.sum(p_new**2)
            new_H = new_U + new_K
            
            # Accept or reject
            accept_prob = min(1.0, torch.exp(current_H - new_H).item())
            
            if np.random.rand() < accept_prob:
                # Accept
                self.unflatten_parameters(q_new)
                samples.append(q_new.clone())
            else:
                # Reject - keep current state
                self.unflatten_parameters(q)
                samples.append(q.clone())
        
        print("✅ HMC completed")
        return torch.stack(samples)

class BayesianEnsemble:
    """
    Ensemble of models for uncertainty quantification
    Implements ensemble sampling and prediction averaging
    """
    
    def __init__(self, model_class, model_args, n_models=10):
        """
        Args:
            model_class: Class of the base model
            model_args: Arguments for model initialization
            n_models: Number of ensemble members
        """
        self.n_models = n_models
        self.models = []
        
        # Initialize ensemble members
        for i in range(n_models):
            model = model_class(**model_args)
            # Add random initialization variations
            for param in model.parameters():
                param.data += 0.01 * torch.randn_like(param.data)
            self.models.append(model)
    
    def train_ensemble(self, train_loader, val_loader, epochs=50, lr=1e-3):
        """Train all ensemble members"""
        print(f"🎯 Training ensemble of {self.n_models} models...")
        
        optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models]
        
        for epoch in range(epochs):
            ensemble_loss = 0
            
            # Train each model
            for model_idx, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                model.train()
                epoch_loss = 0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    
                    mu, log_var = model(X_batch)
                    var = torch.exp(torch.clamp(log_var, min=-4.0, max=1.0))
                    
                    # NLL loss
                    loss = 0.5 * ((y_batch - mu)**2 / var + log_var).mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                ensemble_loss += epoch_loss / len(train_loader)
            
            if epoch % 10 == 0:
                avg_loss = ensemble_loss / self.n_models
                print(f"Epoch {epoch}, Average ensemble loss: {avg_loss:.4f}")
    
    def predict_ensemble(self, X, return_individual=False):
        """
        Make ensemble predictions with uncertainty quantification
        
        Returns:
            mean_pred: Ensemble mean prediction
            aleatoric_uncertainty: Average predictive uncertainty
            epistemic_uncertainty: Model disagreement uncertainty
            total_uncertainty: Combined uncertainty
        """
        predictions = []
        variances = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                mu, log_var = model(X)
                var = torch.exp(torch.clamp(log_var, min=-4.0, max=1.0))
                
                predictions.append(mu)
                variances.append(var)
        
        predictions = torch.stack(predictions)  # [n_models, batch, channels, H, W]
        variances = torch.stack(variances)
        
        # Ensemble statistics
        mean_pred = torch.mean(predictions, dim=0)
        mean_var = torch.mean(variances, dim=0)  # Aleatoric uncertainty
        
        # Epistemic uncertainty (model disagreement)
        epistemic_var = torch.var(predictions, dim=0)
        
        # Total uncertainty
        total_var = mean_var + epistemic_var
        
        if return_individual:
            return mean_pred, mean_var, epistemic_var, total_var, predictions, variances
        else:
            return mean_pred, mean_var, epistemic_var, total_var

def gaussian_likelihood(model, X, y):
    """Gaussian likelihood for climate data"""
    model.eval()
    with torch.no_grad():
        mu, log_var = model(X)
        var = torch.exp(torch.clamp(log_var, min=-4.0, max=1.0))
        
        # Negative log likelihood
        nll = 0.5 * ((y - mu)**2 / var + log_var)
        return -nll.sum().item()  # Return log likelihood

def gaussian_prior(params, sigma=1.0):
    """Gaussian prior on parameters"""
    log_prior = -0.5 * torch.sum(params**2) / (sigma**2)
    return log_prior.item()

def demonstrate_enhanced_mcmc():
    """Demonstration of enhanced MCMC for climate uncertainty"""
    print("🎲 Demonstrating Enhanced MCMC for Climate Uncertainty")
    
    # Create synthetic climate data
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size, channels, H, W = 10, 5, 16, 16
    X = torch.randn(batch_size, channels, H, W)
    y_true = torch.randn(batch_size, 1, H, W)
    
    # Simple model for demonstration
    class SimpleClimateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(5, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
            self.conv3 = nn.Conv2d(8, 2, 3, padding=1)  # mean and log_var
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            mu = x[:, :1, :, :]
            log_var = x[:, 1:, :, :]
            return mu, log_var
    
    model = SimpleClimateModel()
    
    # MCMC sampler
    sampler = MCMCClimateSampler(model, gaussian_likelihood, gaussian_prior)
    
    # Run Metropolis-Hastings (small number for demo)
    samples, log_probs, acceptance_rate = sampler.metropolis_hastings(
        X, y_true, n_samples=100, step_size=0.0001, burn_in=20
    )
    
    # Analyze samples
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trace plot of log probabilities
    axes[0, 0].plot(log_probs)
    axes[0, 0].set_title('MCMC Trace (Log Posterior)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Log Probability')
    
    # Parameter trace (first few parameters)
    n_trace_params = min(5, samples.shape[1])
    for i in range(n_trace_params):
        axes[0, 1].plot(samples[:, i].numpy(), alpha=0.7, label=f'Param {i}')
    axes[0, 1].set_title('Parameter Traces')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].legend()
    
    # Parameter distributions
    param_means = torch.mean(samples, dim=0)
    param_stds = torch.std(samples, dim=0)
    
    axes[1, 0].hist(param_means.numpy(), bins=30, alpha=0.7)
    axes[1, 0].set_title('Distribution of Parameter Means')
    axes[1, 0].set_xlabel('Parameter Mean')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(param_stds.numpy(), bins=30, alpha=0.7)
    axes[1, 1].set_title('Distribution of Parameter Std Devs')
    axes[1, 1].set_xlabel('Parameter Std Dev')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('figures/enhanced_mcmc_demonstration.png', dpi=150, bbox_inches='tight')
    print(f"✅ Enhanced MCMC demonstration saved to figures/enhanced_mcmc_demonstration.png")
    
    # Ensemble demonstration
    print("\n🎯 Demonstrating Bayesian Ensemble...")
    
    ensemble = BayesianEnsemble(
        SimpleClimateModel, {}, n_models=5
    )
    
    # Mock training data
    train_data = [(X[:5], y_true[:5]), (X[5:], y_true[5:])]  # Simple mock
    
    # Make ensemble predictions
    test_X = torch.randn(3, 5, 16, 16)
    mean_pred, aleatoric, epistemic, total = ensemble.predict_ensemble(test_X)
    
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Aleatoric uncertainty: {aleatoric.mean().item():.4f}")
    print(f"Epistemic uncertainty: {epistemic.mean().item():.4f}")
    print(f"Total uncertainty: {total.mean().item():.4f}")
    
    return sampler, ensemble, samples

if __name__ == "__main__":
    # Create figures directory if needed
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Run demonstration
    sampler, ensemble, samples = demonstrate_enhanced_mcmc()
    
    print("\n🎯 Integration with your existing project:")
    print("1. Use MCMCClimateSampler for full Bayesian inference of your U-Net")
    print("2. Apply BayesianEnsemble for robust uncertainty quantification")
    print("3. Use HMC for efficient sampling of high-dimensional parameter spaces")
    print("4. Combine with MRF and HMM for structured uncertainty modeling")