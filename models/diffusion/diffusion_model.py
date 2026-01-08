import torch
import torch.nn as nn

from .sde import ForwardSDE, ReverseSDE
from .wavenet import WaveNet


class DiffusionGenerator(nn.Module):
    """
    Diffusion-based speech generator.
    Combines variance adaptor, WaveNet, and SDE processes.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        wavenet_channels: int = 128,
        wavenet_blocks: int = 20,
        wavenet_kernel_size: int = 3,
        wavenet_dilation_rate: int = 2,
        condition_dim: int = 512,
        speaker_dim: int = 512,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        diffusion_steps: int = 1000,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps

        # SDE processes
        self.forward_sde = ForwardSDE(beta_min, beta_max)
        self.reverse_sde = ReverseSDE(beta_min, beta_max)

        # WaveNet backbone
        self.wavenet = WaveNet(
            latent_dim=latent_dim,
            hidden_dim=wavenet_channels,
            num_blocks=wavenet_blocks,
            kernel_size=wavenet_kernel_size,
            dilation_rate=wavenet_dilation_rate,
            condition_dim=condition_dim,
        )

    def compute_loss(
        self,
        z_0: torch.Tensor,
        p_c: torch.Tensor,
        z_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.

        Args:
            z_0: [B, D, T] ground truth codec latent
            p_c: [B, T, D_cond] content condition
            z_p: [B, T_p, D_spk] speaker prompt

        Returns:
            loss: scalar loss
        """
        batch_size = z_0.shape[0]
        device = z_0.device

        # Sample random time steps
        t = torch.rand(batch_size, device=device)

        # Sample z_t from forward SDE
        z_t, mean, std = self.forward_sde.sample_zt(z_0, t)

        # Predict z_0 from z_t
        z_0_pred = self.wavenet(z_t, t, p_c, z_p)

        # MSE loss
        loss = torch.nn.functional.mse_loss(z_0_pred, z_0)

        return loss

    @torch.no_grad()
    def sample(
        self,
        p_c: torch.Tensor,
        z_p: torch.Tensor,
        num_steps: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample from diffusion model using reverse SDE.

        Args:
            p_c: [B, T, D_cond] content condition
            z_p: [B, T_p, D_spk] speaker prompt
            num_steps: number of reverse diffusion steps
            temperature: sampling temperature

        Returns:
            z_0: [B, D, T] sampled codec latent
        """
        batch_size = p_c.shape[0]
        T_frame = p_c.shape[1]
        device = p_c.device

        # Start from Gaussian noise
        z_t = torch.randn(
            batch_size,
            self.latent_dim,
            T_frame,
            device=device,
        ) * temperature

        # Reverse diffusion
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_value = 1.0 - step * dt
            t = torch.full((batch_size,), t_value, device=device)

            # Predict z_0
            z_0_pred = self.wavenet(z_t, t, p_c, z_p)

            # score ≈ (z_0_pred - z_t) / std^2
            beta_integral = 0.5 * (
                self.forward_sde.beta_min + self.forward_sde.beta_max
            ) * t_value
            beta_integral = torch.tensor(
                beta_integral,
                device=z_t.device,
                dtype=z_t.dtype
            )
            std_squared = 1.0 - torch.exp(-beta_integral)

            score = (z_0_pred - z_t) / std_squared.view(-1, 1, 1)

            # Reverse SDE step
            z_t = self.reverse_sde.reverse_step(z_t, score, t, dt)

        return z_t
