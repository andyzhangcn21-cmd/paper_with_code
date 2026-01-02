import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math


class PGASolver:
    """Projected Gradient Ascent for finding worst-case distribution."""
    
    def __init__(self, model, base_distribution, rho, 
                 lr=0.1, iterations=100, batch_size=32):
        """
        Args:
            model: Pretrained Transformer model
            base_distribution: Base distribution Q0
            rho: Wasserstein radius
            lr: Learning rate for gradient ascent
            iterations: Number of PGA iterations
            batch_size: Batch size for gradient estimation
        """
        self.model = model
        self.base = base_distribution
        self.rho = rho
        self.lr = lr
        self.iterations = iterations
        self.batch_size = batch_size
        self.d = base_distribution.d
        
        # Initialize perturbation parameters
        self.mu = torch.zeros(self.d, requires_grad=True)
        self.sigma = torch.tensor(1.0, requires_grad=True)
    
    def compute_risk_gradient(self, mu, sigma, N_context):
        """Compute gradient of risk w.r.t. distribution parameters."""
        # Enable gradient tracking
        mu = mu.detach().requires_grad_(True)
        sigma = sigma.detach().requires_grad_(True)
        
        # Sample tasks from perturbed distribution
        with torch.no_grad():
            beta = self._sample_from_perturbed(mu, sigma, self.batch_size)
        
        # Sample data and compute loss
        total_loss = 0
        for i in range(self.batch_size):
            X, y = self.base.sample_data(beta[i:i+1], N_context)
            x_test = torch.randn(1, self.d) * self.base.sigma_x
            
            # Get true y for test point
            y_true = x_test @ beta[i].unsqueeze(-1)
            
            # Model prediction
            with torch.no_grad():
                y_pred = self.model(X, y, x_test.squeeze(1))
            
            # Squared loss
            loss = (y_pred - y_true)**2
            total_loss += loss
        
        avg_loss = total_loss / self.batch_size
        
        # Compute gradients
        avg_loss.backward()
        
        return mu.grad, sigma.grad, avg_loss.item()
    
    def _sample_from_perturbed(self, mu, sigma, batch_size):
        """Sample from perturbed distribution N(mu, sigma^2 I)."""
        beta_base = torch.randn(batch_size, self.d)
        beta = beta_base * sigma + mu.expand(batch_size, -1)
        return beta
    
    def project_to_ball(self, mu, sigma):
        """Project distribution parameters to Wasserstein ball."""
        # For isotropic Gaussian: W^2 = ||mu||^2 + d*(sigma - 1)^2
        
        current_dist = torch.norm(mu)**2 + self.d * (sigma - 1)**2
        
        if current_dist <= self.rho**2:
            return mu, sigma
        
        # Need to project
        # We scale both mean and covariance perturbation proportionally
        scale = self.rho / math.sqrt(current_dist.item())
        
        mu_proj = mu * scale
        sigma_proj = 1.0 + (sigma - 1.0) * scale
        
        return mu_proj, sigma_proj
    
    def solve(self, N_context, verbose=True):
        """Run PGA to find worst-case distribution.
        
        Returns:
            mu_adv: Adversarial mean
            sigma_adv: Adversarial standard deviation
            history: Loss history during optimization
        """
        # Initialize parameters
        mu = torch.zeros(self.d)
        sigma = torch.tensor(1.0)
        
        history = []
        
        pbar = tqdm(range(self.iterations), disable=not verbose)
        for iteration in pbar:
            # Compute gradients
            grad_mu, grad_sigma, loss = self.compute_risk_gradient(mu, sigma, N_context)
            
            # Gradient ascent step
            mu = mu + self.lr * grad_mu
            sigma = sigma + self.lr * grad_sigma
            
            # Project to Wasserstein ball
            mu, sigma = self.project_to_ball(mu, sigma)
            
            # Record loss
            history.append(loss)
            
            # Update progress bar
            pbar.set_description(f"Loss: {loss:.4f}, ||mu||: {torch.norm(mu):.3f}, sigma: {sigma:.3f}")
        
        return mu.detach(), sigma.detach(), history
    
    def evaluate_risk(self, mu, sigma, N_context, n_trials=100):
        """Evaluate risk for given distribution parameters."""
        total_loss = 0
        
        for _ in range(n_trials):
            # Sample task
            beta = self._sample_from_perturbed(mu, sigma, 1)
            
            # Sample context data
            X, y = self.base.sample_data(beta, N_context)
            
            # Sample test point
            x_test = torch.randn(1, self.d) * self.base.sigma_x
            y_true = x_test @ beta.T
            
            # Model prediction
            with torch.no_grad():
                y_pred = self.model(X, y, x_test.squeeze(1))
            
            # Loss
            loss = (y_pred - y_true)**2
            total_loss += loss.item()
        
        return total_loss / n_trials