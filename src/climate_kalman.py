# src/climate_kalman.py
# Kalman Filter for Climate State Estimation
# Implements linear and Extended Kalman Filters for climate time series

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.linalg import inv, cholesky
import seaborn as sns

class ClimateKalmanFilter:
    """
    Kalman Filter for climate state estimation and prediction
    
    Key concepts from syllabus:
    - State space models
    - Linear Gaussian systems
    - Optimal filtering and prediction
    - Forward-backward smoothing
    - Handling missing observations
    
    State model: x_t = F*x_{t-1} + B*u_t + w_t,  w_t ~ N(0, Q)
    Observation: y_t = H*x_t + v_t,              v_t ~ N(0, R)
    """
    
    def __init__(self, dim_state, dim_obs, dim_control=0):
        """
        Args:
            dim_state: Dimension of hidden climate state
            dim_obs: Dimension of observations (e.g., temperature measurements)
            dim_control: Dimension of control inputs (optional)
        """
        self.dim_x = dim_state
        self.dim_z = dim_obs
        self.dim_u = dim_control
        
        # State transition matrix F (dynamics)
        self.F = np.eye(dim_state)
        
        # Control matrix B
        self.B = np.zeros((dim_state, dim_control)) if dim_control > 0 else None
        
        # Observation matrix H
        self.H = np.eye(dim_obs, dim_state)
        
        # Process noise covariance Q
        self.Q = 0.1 * np.eye(dim_state)
        
        # Observation noise covariance R
        self.R = 0.1 * np.eye(dim_obs)
        
        # Initial state and covariance
        self.x = np.zeros(dim_state)  # State estimate
        self.P = np.eye(dim_state)    # Error covariance
        
        # Storage for filtering results
        self.history = {
            'x_prior': [],
            'P_prior': [],
            'x_post': [],
            'P_post': [],
            'innovation': [],
            'S': [],
            'K': [],
            'log_likelihood': []
        }
    
    def predict(self, u=None):
        """
        Prediction step (time update)
        
        x̂_{k|k-1} = F * x̂_{k-1|k-1} + B * u_k
        P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        """
        # State prediction
        if u is not None and self.B is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Store prior estimates
        self.history['x_prior'].append(self.x.copy())
        self.history['P_prior'].append(self.P.copy())
    
    def update(self, z, R=None):
        """
        Update step (measurement update)
        
        Innovation: ỹ_k = z_k - H * x̂_{k|k-1}
        Innovation covariance: S_k = H * P_{k|k-1} * H^T + R
        Kalman gain: K_k = P_{k|k-1} * H^T * S_k^{-1}
        State update: x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k
        Covariance update: P_{k|k} = (I - K_k * H) * P_{k|k-1}
        """
        if R is None:
            R = self.R
        
        # Innovation (residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman gain
        K = self.P @ self.H.T @ inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Compute log likelihood of observation
        log_likelihood = -0.5 * (y.T @ inv(S) @ y + np.log(np.linalg.det(2 * np.pi * S)))
        
        # Store results
        self.history['innovation'].append(y.copy())
        self.history['S'].append(S.copy())
        self.history['K'].append(K.copy())
        self.history['x_post'].append(self.x.copy())
        self.history['P_post'].append(self.P.copy())
        self.history['log_likelihood'].append(log_likelihood)
        
        return y, S, K
    
    def filter_sequence(self, observations, controls=None, missing_mask=None):
        """
        Filter a complete sequence of observations
        
        Args:
            observations: Array of shape (T, dim_obs)
            controls: Array of shape (T, dim_control) or None
            missing_mask: Boolean array (T,) indicating missing observations
        """
        T = len(observations)
        
        # Initialize storage
        x_filtered = np.zeros((T, self.dim_x))
        P_filtered = np.zeros((T, self.dim_x, self.dim_x))
        
        for t in range(T):
            # Predict step
            u_t = controls[t] if controls is not None else None
            self.predict(u_t)
            
            # Update step (if observation is available)
            if missing_mask is None or not missing_mask[t]:
                self.update(observations[t])
            
            # Store results
            x_filtered[t] = self.x.copy()
            P_filtered[t] = self.P.copy()
        
        return x_filtered, P_filtered
    
    def smooth_rts(self, x_filtered, P_filtered):
        """
        Rauch-Tung-Striebel (RTS) smoother for optimal smoothing
        
        Backward pass:
        x̂_{k|T} = x̂_{k|k} + A_k * (x̂_{k+1|T} - F * x̂_{k|k})
        P_{k|T} = P_{k|k} + A_k * (P_{k+1|T} - P_{k+1|k}) * A_k^T
        
        where A_k = P_{k|k} * F^T * P_{k+1|k}^{-1}
        """
        T = len(x_filtered)
        
        # Initialize with filtered estimates
        x_smooth = x_filtered.copy()
        P_smooth = P_filtered.copy()
        
        # Backward pass
        for t in range(T-2, -1, -1):
            # Predicted state at t+1
            x_pred = self.F @ x_filtered[t]
            P_pred = self.F @ P_filtered[t] @ self.F.T + self.Q
            
            # Smoother gain
            A = P_filtered[t] @ self.F.T @ inv(P_pred)
            
            # Smoothed estimates
            x_smooth[t] = x_filtered[t] + A @ (x_smooth[t+1] - x_pred)
            P_smooth[t] = P_filtered[t] + A @ (P_smooth[t+1] - P_pred) @ A.T
        
        return x_smooth, P_smooth

class ExtendedKalmanFilter(ClimateKalmanFilter):
    """
    Extended Kalman Filter for nonlinear climate dynamics
    
    For nonlinear systems:
    x_t = f(x_{t-1}, u_t) + w_t
    y_t = h(x_t) + v_t
    
    Uses linearization around current state estimate
    """
    
    def __init__(self, dim_state, dim_obs, dim_control=0):
        super().__init__(dim_state, dim_obs, dim_control)
        
        # Nonlinear functions (to be overridden)
        self.f = None  # State transition function
        self.h = None  # Observation function
        self.jac_f = None  # Jacobian of f
        self.jac_h = None  # Jacobian of h
    
    def predict_nonlinear(self, u=None):
        """Prediction step for nonlinear dynamics"""
        if self.f is None:
            return super().predict(u)
        
        # Nonlinear state prediction
        self.x = self.f(self.x, u)
        
        # Linearize around predicted state
        F_k = self.jac_f(self.x, u)
        
        # Covariance prediction
        self.P = F_k @ self.P @ F_k.T + self.Q
        
        # Store prior estimates
        self.history['x_prior'].append(self.x.copy())
        self.history['P_prior'].append(self.P.copy())
    
    def update_nonlinear(self, z, R=None):
        """Update step for nonlinear observations"""
        if self.h is None:
            return super().update(z, R)
        
        if R is None:
            R = self.R
        
        # Nonlinear observation prediction
        z_pred = self.h(self.x)
        
        # Linearize around current state
        H_k = self.jac_h(self.x)
        
        # Innovation
        y = z - z_pred
        
        # Innovation covariance
        S = H_k @ self.P @ H_k.T + R
        
        # Kalman gain
        K = self.P @ H_k.T @ inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I_KH = np.eye(self.dim_x) - K @ H_k
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Log likelihood
        log_likelihood = -0.5 * (y.T @ inv(S) @ y + np.log(np.linalg.det(2 * np.pi * S)))
        
        # Store results
        self.history['innovation'].append(y.copy())
        self.history['S'].append(S.copy())
        self.history['K'].append(K.copy())
        self.history['x_post'].append(self.x.copy())
        self.history['P_post'].append(self.P.copy())
        self.history['log_likelihood'].append(log_likelihood)
        
        return y, S, K

class NeuralKalmanFilter(nn.Module):
    """
    Neural network-enhanced Kalman Filter
    Learns nonlinear dynamics and observation models from data
    """
    
    def __init__(self, dim_state, dim_obs, hidden_dim=64):
        super().__init__()
        
        self.dim_x = dim_state
        self.dim_z = dim_obs
        
        # Neural networks for dynamics and observations
        self.dynamics_net = nn.Sequential(
            nn.Linear(dim_state, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_state)
        )
        
        self.observation_net = nn.Sequential(
            nn.Linear(dim_state, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_obs)
        )
        
        # Learnable noise parameters
        self.log_Q_diag = nn.Parameter(torch.zeros(dim_state))
        self.log_R_diag = nn.Parameter(torch.zeros(dim_obs))
        
        # Initial state parameters
        self.x0 = nn.Parameter(torch.zeros(dim_state))
        self.log_P0_diag = nn.Parameter(torch.zeros(dim_state))
    
    def dynamics(self, x):
        """Learned dynamics function"""
        return self.dynamics_net(x)
    
    def observation(self, x):
        """Learned observation function"""
        return self.observation_net(x)
    
    def forward(self, observations, return_likelihood=False):
        """
        Forward pass through the sequence using learned Kalman filtering
        """
        batch_size, seq_len, _ = observations.shape
        device = observations.device
        
        # Initialize state
        x = self.x0.unsqueeze(0).repeat(batch_size, 1)
        P = torch.diag_embed(torch.exp(self.log_P0_diag)).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Noise covariances
        Q = torch.diag_embed(torch.exp(self.log_Q_diag))
        R = torch.diag_embed(torch.exp(self.log_R_diag))
        
        filtered_states = []
        log_likelihoods = []
        
        for t in range(seq_len):
            # Predict step
            x_pred = self.dynamics(x)
            
            # Linearize dynamics (compute Jacobian)
            F = self._compute_jacobian(self.dynamics_net, x)
            P_pred = torch.bmm(torch.bmm(F, P), F.transpose(-1, -2)) + Q.unsqueeze(0)
            
            # Update step
            z_pred = self.observation(x_pred)
            
            # Linearize observation
            H = self._compute_jacobian(self.observation_net, x_pred)
            
            # Innovation
            y = observations[:, t, :] - z_pred
            S = torch.bmm(torch.bmm(H, P_pred), H.transpose(-1, -2)) + R.unsqueeze(0)
            
            # Kalman gain
            K = torch.bmm(torch.bmm(P_pred, H.transpose(-1, -2)), torch.inverse(S))
            
            # State update
            x = x_pred + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
            
            # Covariance update
            I_KH = torch.eye(self.dim_x, device=device).unsqueeze(0) - torch.bmm(K, H)
            P = torch.bmm(I_KH, P_pred)
            
            # Log likelihood
            if return_likelihood:
                log_det_S = torch.logdet(2 * np.pi * S)
                mahalanobis = torch.bmm(y.unsqueeze(-2), torch.bmm(torch.inverse(S), y.unsqueeze(-1)))
                log_lik = -0.5 * (mahalanobis.squeeze() + log_det_S)
                log_likelihoods.append(log_lik)
            
            filtered_states.append(x)
        
        filtered_states = torch.stack(filtered_states, dim=1)
        
        if return_likelihood:
            total_log_likelihood = torch.stack(log_likelihoods, dim=1).sum(dim=1)
            return filtered_states, total_log_likelihood
        
        return filtered_states
    
    def _compute_jacobian(self, func, x):
        """Compute Jacobian matrix using automatic differentiation"""
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        # Use torch.autograd.functional.jacobian for batch computation
        def single_func(xi):
            return func(xi.unsqueeze(0)).squeeze(0)
        
        jacobians = []
        for i in range(batch_size):
            jac = torch.autograd.functional.jacobian(single_func, x[i])
            jacobians.append(jac)
        
        return torch.stack(jacobians)

def demonstrate_climate_kalman():
    """Demonstration of Kalman filtering for climate data"""
    print("📊 Demonstrating Kalman Filter for Climate State Estimation")
    
    # Generate synthetic climate time series
    np.random.seed(42)
    T = 100  # Time steps
    
    # True climate system parameters
    dim_state = 3  # [temperature_trend, seasonal_component, random_walk]
    dim_obs = 2    # [surface_temp, sea_surface_temp]
    
    # True dynamics (AR model with seasonal component)
    F_true = np.array([
        [0.95, 0.0, 0.1],   # Temperature trend with some coupling
        [0.0, 0.9, 0.0],    # Seasonal component  
        [0.0, 0.0, 0.8]     # Random walk component
    ])
    
    # True observation model
    H_true = np.array([
        [1.0, 1.0, 0.5],    # Surface temp = trend + seasonal + 0.5*random
        [0.8, 0.5, 0.2]     # Sea surface temp = 0.8*trend + 0.5*seasonal + 0.2*random
    ])
    
    # Generate true states and observations
    Q_true = 0.1 * np.eye(dim_state)
    R_true = 0.2 * np.eye(dim_obs)
    
    x_true = np.zeros((T, dim_state))
    y_obs = np.zeros((T, dim_obs))
    
    x_true[0] = np.random.randn(dim_state)
    
    for t in range(1, T):
        # State evolution
        x_true[t] = F_true @ x_true[t-1] + np.random.multivariate_normal(np.zeros(dim_state), Q_true)
        
        # Observations
        y_obs[t] = H_true @ x_true[t] + np.random.multivariate_normal(np.zeros(dim_obs), R_true)
    
    # Initialize Kalman filter
    kf = ClimateKalmanFilter(dim_state, dim_obs)
    
    # Configure with approximate parameters (in practice, these would be learned)
    kf.F = np.array([
        [0.9, 0.0, 0.05],
        [0.0, 0.85, 0.0],
        [0.0, 0.0, 0.75]
    ])
    
    kf.H = np.array([
        [1.0, 0.9, 0.4],
        [0.7, 0.6, 0.3]
    ])
    
    kf.Q = 0.15 * np.eye(dim_state)
    kf.R = 0.25 * np.eye(dim_obs)
    
    # Initial state estimate
    kf.x = np.zeros(dim_state)
    kf.P = np.eye(dim_state)
    
    # Filter the sequence
    x_filtered, P_filtered = kf.filter_sequence(y_obs)
    
    # Apply RTS smoother
    x_smooth, P_smooth = kf.smooth_rts(x_filtered, P_filtered)
    
    # Visualize results
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    time_points = np.arange(T)
    
    # State components
    state_names = ['Temperature Trend', 'Seasonal Component', 'Random Walk']
    colors = ['blue', 'red', 'green']
    
    for i in range(dim_state):
        # Filtered vs true states
        axes[i, 0].plot(time_points, x_true[:, i], 'k-', label='True State', linewidth=2)
        axes[i, 0].plot(time_points, x_filtered[:, i], colors[i] + '--', label='Filtered', linewidth=1.5)
        axes[i, 0].plot(time_points, x_smooth[:, i], colors[i] + ':', label='Smoothed', linewidth=1.5)
        
        # Uncertainty bands
        std_filtered = np.sqrt(P_filtered[:, i, i])
        axes[i, 0].fill_between(time_points, 
                               x_filtered[:, i] - 2*std_filtered,
                               x_filtered[:, i] + 2*std_filtered,
                               alpha=0.2, color=colors[i])
        
        axes[i, 0].set_title(f'{state_names[i]} - State Estimation')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Estimation errors
        error_filtered = x_filtered[:, i] - x_true[:, i]
        error_smooth = x_smooth[:, i] - x_true[:, i]
        
        axes[i, 1].plot(time_points, error_filtered, colors[i] + '--', label='Filtered Error')
        axes[i, 1].plot(time_points, error_smooth, colors[i] + ':', label='Smoothed Error')
        axes[i, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[i, 1].set_title(f'{state_names[i]} - Estimation Errors')
        axes[i, 1].set_ylabel('Error')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Time')
    axes[2, 1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('figures/kalman_demonstration.png', dpi=150, bbox_inches='tight')
    print(f"✅ Kalman filter demonstration saved to figures/kalman_demonstration.png")
    
    # Compute performance metrics
    rmse_filtered = np.sqrt(np.mean((x_filtered - x_true)**2, axis=0))
    rmse_smooth = np.sqrt(np.mean((x_smooth - x_true)**2, axis=0))
    
    print("\n📊 Performance Metrics:")
    for i, name in enumerate(state_names):
        print(f"{name}:")
        print(f"  Filtered RMSE: {rmse_filtered[i]:.4f}")
        print(f"  Smoothed RMSE: {rmse_smooth[i]:.4f}")
    
    # Total log likelihood
    total_log_likelihood = sum(kf.history['log_likelihood'])
    print(f"\nTotal log likelihood: {total_log_likelihood:.2f}")
    
    return kf, x_true, x_filtered, x_smooth

if __name__ == "__main__":
    # Create figures directory if needed
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Run demonstration
    kf, x_true, x_filtered, x_smooth = demonstrate_climate_kalman()
    
    print("\n🎯 Integration with your existing project:")
    print("1. Use ClimateKalmanFilter for temporal smoothing of climate predictions")
    print("2. Apply ExtendedKalmanFilter for nonlinear climate dynamics")
    print("3. Use NeuralKalmanFilter to learn dynamics from your ERA5 data")
    print("4. Combine with HMM for regime-dependent Kalman filtering")
    print("5. Handle missing climate observations with the filtering framework")