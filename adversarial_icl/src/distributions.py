import torch
import numpy as np
from scipy.stats import ortho_group
import math


class TaskDistribution:
    """Base class for generating tasks from distributions."""
    
    def sample_task(self, batch_size=1):
        """Sample a batch of tasks.
        
        Args:
            batch_size: Number of tasks to sample
        
        Returns:
            beta: Task parameters [batch_size, d]
        """
        raise NotImplementedError
    
    def sample_data(self, beta, N, sigma=0.1):
        """Sample data for given tasks.
        
        Args:
            beta: Task parameters [batch_size, d]
            N: Number of data points per task
            sigma: Noise standard deviation
        
        Returns:
            X: Inputs [batch_size, N, d]
            y: Outputs [batch_size, N, 1]
        """
        raise NotImplementedError


class GaussianTaskDistribution(TaskDistribution):
    """Gaussian task distribution."""
    
    def __init__(self, d, mu_beta=0.0, sigma_beta=1.0, sigma_x=1.0, seed=None):
        """
        Args:
            d: Input dimension
            mu_beta: Mean of beta distribution
            sigma_beta: Standard deviation of beta distribution
            sigma_x: Standard deviation of input x
        """
        self.d = d
        self.mu_beta = mu_beta
        self.sigma_beta = sigma_beta
        self.sigma_x = sigma_x
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def sample_task(self, batch_size=1):
        """Sample beta from Gaussian distribution."""
        beta = torch.randn(batch_size, self.d) * self.sigma_beta + self.mu_beta
        return beta
    
    def sample_data(self, beta, N, sigma=0.1):
        """Sample data for linear regression tasks."""
        batch_size = beta.shape[0]
        
        # Sample inputs
        X = torch.randn(batch_size, N, self.d) * self.sigma_x
        
        # Generate outputs with noise
        y = torch.einsum('bnd,bd->bn', X, beta).unsqueeze(-1)
        noise = torch.randn_like(y) * sigma
        y = y + noise
        
        return X, y


class WassersteinPerturbedDistribution:
    """Create adversarially perturbed distributions within Wasserstein ball."""
    
    def __init__(self, base_distribution, rho, p=2):
        """
        Args:
            base_distribution: Base task distribution
            rho: Wasserstein radius
            p: Order of Wasserstein distance
        """
        self.base = base_distribution
        self.rho = rho
        self.p = p
        self.d = base_distribution.d
    
    def sample_perturbed(self, mu_perturb=None, sigma_perturb=None, batch_size=1):
        """Sample from perturbed distribution.
        
        Args:
            mu_perturb: Perturbation to mean (None for random within ball)
            sigma_perturb: Perturbation to covariance (None for random within ball)
            batch_size: Number of tasks to sample
        
        Returns:
            beta: Task parameters from perturbed distribution
        """
        if mu_perturb is None or sigma_perturb is None:
            # Sample random perturbation within Wasserstein ball
            mu_perturb, sigma_perturb = self._sample_random_perturbation()
        
        # Sample from base distribution and apply perturbation
        beta_base = self.base.sample_task(batch_size)
        
        # Apply mean shift
        beta = beta_base + mu_perturb.expand(batch_size, -1)
        
        # Apply covariance scaling (simplified)
        if sigma_perturb != 1.0:
            beta = beta * sigma_perturb
        
        return beta
    
    def _sample_random_perturbation(self):
        """Sample random perturbation within Wasserstein ball."""
        # For isotropic Gaussian, Wasserstein distance is:
        # W^2 = ||mu||^2 + d*(sigma - 1)^2
        
        # Sample random direction for mean
        mu_dir = torch.randn(self.d)
        mu_dir = mu_dir / torch.norm(mu_dir)
        
        # Sample random sigma perturbation
        # We need to satisfy: ||mu||^2 + d*(sigma - 1)^2 <= rho^2
        
        # Randomly allocate budget between mean and covariance
        alpha = torch.rand(1).item()  # fraction for mean
        mu_norm = math.sqrt(alpha * self.rho**2)
        sigma_diff = math.sqrt((1 - alpha) * self.rho**2 / self.d)
        
        mu_perturb = mu_dir * mu_norm
        sigma_perturb = 1.0 + torch.randn(1).item() * sigma_diff
        
        return mu_perturb, sigma_perturb.item()
    
    def wasserstein_distance(self, mu1, sigma1, mu2, sigma2):
        """Compute Wasserstein-2 distance between two isotropic Gaussians."""
        # For isotropic Gaussians: W^2 = ||mu1 - mu2||^2 + d*(sigma1 - sigma2)^2
        mean_dist = torch.norm(mu1 - mu2)**2
        var_dist = self.d * (sigma1 - sigma2)**2
        return math.sqrt(mean_dist + var_dist)