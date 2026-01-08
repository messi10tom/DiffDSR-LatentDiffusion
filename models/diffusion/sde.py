import torch


class ForwardSDE:
    """
    Forward SDE process (Eq. 2 in paper).
    dz_t = -0.5 * β_t * z_t * dt + sqrt(β_t) * dw_t
    """
    
    def __init__(
        self,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        T: float = 1.0
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Linear noise schedule."""
        return self.beta_min + (self.beta_max - self.beta_min) * t
    
    def forward_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        dt: float = 0.001
    ) -> torch.Tensor:
        """
        Single forward step.
        
        Args:
            z_t: [B, D, T] current state
            t: [B] current time
            dt: time step
        
        Returns:
            z_t_plus_dt: [B, D, T] next state
        """
        beta = self.beta_t(t).view(-1, 1, 1)
        
        drift = -0.5 * beta * z_t * dt
        diffusion = torch.sqrt(beta * dt) * torch.randn_like(z_t)
        
        z_t_plus_dt = z_t + drift + diffusion
        
        return z_t_plus_dt
    
    def sample_zt(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor
    ) -> tuple:
        """
        Sample z_t from z_0 directly.
        
        Args:
            z_0: [B, D, T] initial state
            t: [B] time points
        
        Returns:
            z_t: [B, D, T] noisy state at time t
            mean: [B, D, T] mean
            std: [B, D, T] standard deviation
        """
        # Compute integral of beta from 0 to t
        beta_integral = 0.5 * (self.beta_min + self.beta_max) * t
        beta_integral = beta_integral.view(-1, 1, 1)
        
        # Mean and std
        mean = z_0 * torch.exp(-0.5 * beta_integral)
        std = torch.sqrt(1 - torch.exp(-beta_integral))
        
        # Sample z_t
        noise = torch.randn_like(z_0)
        z_t = mean + std * noise
        
        return z_t, mean, std


class ReverseSDE:
    """
    Reverse SDE process (Eq. 3 in paper).
    dz_t = -0.5 * (z_t + ∇log p_t(z_t)) * β_t * dt
    """
    
    def __init__(
        self,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        T: float = 1.0
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Linear noise schedule."""
        return self.beta_min + (self.beta_max - self.beta_min) * t
    
    def reverse_step(
        self,
        z_t: torch.Tensor,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: float = 0.001
    ) -> torch.Tensor:
        """
        Single reverse step.
        
        Args:
            z_t: [B, D, T] current state
            score: [B, D, T] score function ∇log p_t(z_t)
            t: [B] current time
            dt: time step
        
        Returns:
            z_t_minus_dt: [B, D, T] previous state
        """
        beta = self.beta_t(t).view(-1, 1, 1)
        
        drift = -0.5 * (z_t + score) * beta * dt
        
        z_t_minus_dt = z_t + drift
        
        return z_t_minus_dt
