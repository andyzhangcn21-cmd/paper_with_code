import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import math

from .linear_transformer import LinearTransformerICL
from .distributions import GaussianTaskDistribution
from .wasserstein_search import PGASolver


class ExperimentRunner:
    """Runner for all experiments."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def setup_model(self, d, m, n_heads=1):
        """Initialize and pretrain a linear Transformer."""
        # Model dimension matches input dimension + 1 (for y)
        d_model = d + 1
        
        model = LinearTransformerICL(
            d_model=d_model,
            n_heads=n_heads,
            d_k=m,  # Attention head dimension = capacity parameter
            output_dim=1
        ).to(self.device)
        
        return model
    
    def pretrain_model(self, model, d, pretrain_steps=10000, batch_size=32, 
                       N_context=10, lr=1e-3):
        """Pretrain model on linear regression tasks."""
        # Create pretraining distribution
        pretrain_dist = GaussianTaskDistribution(d)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=pretrain_steps)
        
        losses = []
        pbar = tqdm(range(pretrain_steps), desc="Pretraining")
        
        for step in pbar:
            # Sample task
            beta = pretrain_dist.sample_task(batch_size)
            
            # Sample context data
            X, y = pretrain_dist.sample_data(beta, N_context, sigma=0.1)
            
            # Sample test point
            x_test = torch.randn(batch_size, d) * pretrain_dist.sigma_x
            y_true = torch.einsum('bd,bd->b', x_test, beta).unsqueeze(-1)
            
            # Move to device
            X, y, x_test, y_true = [t.to(self.device) for t in [X, y, x_test, y_true]]
            
            # Model prediction
            y_pred = model(X, y, x_test)
            
            # Loss
            loss = nn.MSELoss()(y_pred, y_true)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': np.mean(losses[-100:])})
        
        return model, losses
    
    def experiment1_risk_vs_radius(self, d=20, m=16, N=15):
        """Experiment 1: Risk growth with adversarial radius."""
        print("Running Experiment 1: Risk vs Radius")
        
        # Setup and pretrain model
        model = self.setup_model(d, m)
        model, _ = self.pretrain_model(model, d)
        model.eval()
        
        # Base distribution
        base_dist = GaussianTaskDistribution(d)
        
        # Test different radii
        radii = np.linspace(0, 2.0, 11)
        risks = []
        
        for rho in tqdm(radii, desc="Testing radii"):
            # Find worst-case distribution
            solver = PGASolver(
                model=model,
                base_distribution=base_dist,
                rho=rho,
                lr=0.05,
                iterations=50,
                batch_size=16
            )
            
            mu_adv, sigma_adv, _ = solver.solve(N_context=N, verbose=False)
            
            # Evaluate risk
            risk = solver.evaluate_risk(mu_adv, sigma_adv, N_context=N, n_trials=50)
            risks.append(risk)
        
        # Fit quadratic curve
        def quadratic_func(r, a, b):
            return a * r + b * r**2
        
        try:
            popt, _ = curve_fit(quadratic_func, radii, risks, p0=[0.1, 0.1])
            a_fit, b_fit = popt
            fitted_curve = quadratic_func(radii, a_fit, b_fit)
            r_squared = 1 - np.sum((risks - fitted_curve)**2) / np.sum((risks - np.mean(risks))**2)
            
            print(f"Fit: a={a_fit:.4f}, b={b_fit:.4f}, R²={r_squared:.4f}")
        except:
            a_fit, b_fit = 0, 0
            fitted_curve = None
            r_squared = 0
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(radii, risks, alpha=0.7, label='Experimental data')
        if fitted_curve is not None:
            ax.plot(radii, fitted_curve, 'r-', label=f'Quadratic fit: a={a_fit:.3f}, b={b_fit:.3f}')
        ax.set_xlabel('Adversarial Radius ρ', fontsize=12)
        ax.set_ylabel('Worst-case Risk', fontsize=12)
        ax.set_title('Experiment 1: Risk Growth with Adversarial Radius', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiment1_risk_vs_radius.png', dpi=150)
        plt.show()
        
        return radii, risks, (a_fit, b_fit, r_squared)
    
    def experiment2_safe_radius_vs_capacity(self, d=20, N=15, epsilon=0.5):
        """Experiment 2: Safe radius vs model capacity."""
        print("\nRunning Experiment 2: Safe Radius vs Capacity")
        
        capacities = [4, 8, 16, 32, 64]
        safe_radii = []
        
        for m in tqdm(capacities, desc="Testing capacities"):
            # Setup and pretrain model
            model = self.setup_model(d, m)
            model, _ = self.pretrain_model(model, d)
            model.eval()
            
            # Base distribution
            base_dist = GaussianTaskDistribution(d)
            
            # Find safe radius via binary search
            low, high = 0.0, 3.0
            tolerance = 0.05
            
            while high - low > tolerance:
                rho_mid = (low + high) / 2
                
                # Find worst-case distribution for this radius
                solver = PGASolver(
                    model=model,
                    base_distribution=base_dist,
                    rho=rho_mid,
                    lr=0.05,
                    iterations=30,
                    batch_size=16
                )
                
                mu_adv, sigma_adv, _ = solver.solve(N_context=N, verbose=False)
                
                # Evaluate nominal risk
                risk_nominal = self._evaluate_nominal_risk(model, base_dist, N, n_trials=50)
                
                # Evaluate adversarial risk
                risk_adv = solver.evaluate_risk(mu_adv, sigma_adv, N_context=N, n_trials=30)
                
                risk_increment = risk_adv - risk_nominal
                
                if risk_increment <= epsilon:
                    low = rho_mid  # Can tolerate larger radius
                else:
                    high = rho_mid  # Need smaller radius
            
            safe_radius = (low + high) / 2
            safe_radii.append(safe_radius)
            
            print(f"Capacity m={m}: safe_radius={safe_radius:.3f}")
        
        # Fit sqrt(m) relationship
        sqrt_capacities = np.sqrt(capacities)
        
        # Linear regression
        A = np.vstack([sqrt_capacities, np.ones(len(capacities))]).T
        slope, intercept = np.linalg.lstsq(A, safe_radii, rcond=None)[0]
        
        fitted_line = slope * sqrt_capacities + intercept
        
        # Compute R²
        ss_res = np.sum((safe_radii - fitted_line)**2)
        ss_tot = np.sum((safe_radii - np.mean(safe_radii))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sqrt_capacities, safe_radii, alpha=0.7, s=100, 
                  label='Experimental data')
        ax.plot(sqrt_capacities, fitted_line, 'r--', 
                label=f'Linear fit: slope={slope:.3f}, R²={r_squared:.4f}')
        ax.set_xlabel('√m (Square root of model capacity)', fontsize=12)
        ax.set_ylabel('Safe Radius ρ_max', fontsize=12)
        ax.set_title('Experiment 2: Safe Radius vs Model Capacity', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiment2_safe_radius_vs_capacity.png', dpi=150)
        plt.show()
        
        return capacities, safe_radii, (slope, intercept, r_squared)
    
    def experiment3_sample_tax(self, d=20, m=16, N0=5):
        """Experiment 3: Sample complexity tax."""
        print("\nRunning Experiment 3: Sample Complexity Tax")
        
        # Setup and pretrain model
        model = self.setup_model(d, m)
        model, _ = self.pretrain_model(model, d)
        model.eval()
        
        # Base distribution
        base_dist = GaussianTaskDistribution(d)
        
        # Find target risk level with N0 examples
        target_risk = self._evaluate_nominal_risk(model, base_dist, N0, n_trials=100)
        print(f"Target risk with N0={N0}: {target_risk:.4f}")
        
        # Test different radii
        radii = np.linspace(0, 1.5, 8)
        required_samples = []
        
        for rho in tqdm(radii, desc="Testing radii for sample tax"):
            # Binary search for required N
            N_low, N_high = N0, N0 * 10
            tolerance = 1
            
            while N_high - N_low > tolerance:
                N_mid = int((N_low + N_high) / 2)
                
                # Find worst-case distribution
                solver = PGASolver(
                    model=model,
                    base_distribution=base_dist,
                    rho=rho,
                    lr=0.05,
                    iterations=30,
                    batch_size=16
                )
                
                mu_adv, sigma_adv, _ = solver.solve(N_context=N_mid, verbose=False)
                
                # Evaluate adversarial risk
                risk_adv = solver.evaluate_risk(mu_adv, sigma_adv, 
                                                N_context=N_mid, n_trials=30)
                
                if risk_adv <= target_risk:
                    N_high = N_mid  # Can achieve with fewer samples
                else:
                    N_low = N_mid  # Need more samples
            
            required_N = int((N_low + N_high) / 2)
            required_samples.append(required_N)
            
            print(f"Radius ρ={rho:.2f}: requires N={required_N} samples")
        
        # Compute extra samples
        extra_samples = np.array(required_samples) - N0
        radii_squared = radii**2
        
        # Linear fit for extra_samples vs ρ²
        A = np.vstack([radii_squared, np.ones(len(radii))]).T
        slope, intercept = np.linalg.lstsq(A, extra_samples, rcond=None)[0]
        
        fitted_line = slope * radii_squared + intercept
        
        # Compute R²
        ss_res = np.sum((extra_samples - fitted_line)**2)
        ss_tot = np.sum((extra_samples - np.mean(extra_samples))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(radii, required_samples, alpha=0.7, s=100)
        ax1.set_xlabel('Adversarial Radius ρ', fontsize=12)
        ax1.set_ylabel('Required Samples N_ρ', fontsize=12)
        ax1.set_title('Required Samples vs Radius', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(radii_squared, extra_samples, alpha=0.7, s=100, 
                   label='Experimental data')
        ax2.plot(radii_squared, fitted_line, 'r--', 
                label=f'Linear fit: slope={slope:.2f}, R²={r_squared:.4f}')
        ax2.set_xlabel('ρ² (Squared radius)', fontsize=12)
        ax2.set_ylabel('Extra Samples (N_ρ - N₀)', fontsize=12)
        ax2.set_title('Extra Samples vs ρ²', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Experiment 3: Sample Complexity Tax', fontsize=16)
        plt.tight_layout()
        plt.savefig('experiment3_sample_tax.png', dpi=150)
        plt.show()
        
        return radii, required_samples, extra_samples, (slope, intercept, r_squared)
    
    def _evaluate_nominal_risk(self, model, distribution, N, n_trials=100):
        """Evaluate model risk under nominal distribution."""
        total_loss = 0
        
        for _ in range(n_trials):
            # Sample task from nominal distribution
            beta = distribution.sample_task(1)
            
            # Sample context data
            X, y = distribution.sample_data(beta, N, sigma=0.1)
            
            # Sample test point
            d = distribution.d
            x_test = torch.randn(1, d) * distribution.sigma_x
            y_true = x_test @ beta.T
            
            # Move to device
            X, y, x_test, y_true = [t.to(self.device) for t in [X, y, x_test, y_true]]
            
            # Model prediction
            with torch.no_grad():
                y_pred = model(X, y, x_test.squeeze(1))
            
            # Loss
            loss = (y_pred - y_true)**2
            total_loss += loss.item()
        
        return total_loss / n_trials