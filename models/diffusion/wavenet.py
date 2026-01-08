import torch
import torch.nn as nn
import math


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, condition_dim: int, hidden_dim: int):
        super().__init__()
        
        self.scale_transform = nn.Linear(condition_dim, hidden_dim)
        self.shift_transform = nn.Linear(condition_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, T] features
            condition: [B, D_cond] or [B, T, D_cond] condition
        
        Returns:
            modulated: [B, D, T] modulated features
        """
        if condition.dim() == 3:
            # Average pool over time
            condition = condition.mean(dim=1)
        
        scale = self.scale_transform(condition).unsqueeze(-1)  # [B, D, 1]
        shift = self.shift_transform(condition).unsqueeze(-1)  # [B, D, 1]
        
        return x * scale + shift


class WaveNetBlock(nn.Module):
    """
    WaveNet block with:
    - Dilated convolution
    - Q-K-V attention
    - FiLM conditioning
    """
    
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        condition_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Dilated convolution
        self.dilated_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2
        )
        
        # Q-K-V attention with speaker prompt
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FiLM conditioning
        self.film_condition = FiLMLayer(condition_dim, hidden_dim)
        self.film_time = FiLMLayer(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Residual projection
        self.residual_projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        time_emb: torch.Tensor,
        speaker_prompt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, D, T] input features
            condition: [B, T, D_cond] content condition p_c
            time_emb: [B, D] time embedding
            speaker_prompt: [B, T_p, D_spk] speaker prompt z_p
        
        Returns:
            output: [B, D, T] output features
        """
        residual = x
        
        # Dilated convolution
        conv_out = self.dilated_conv(x)  # [B, 2D, T]
        gate, filter_out = conv_out.chunk(2, dim=1)  # [B, D, T] each
        conv_out = torch.sigmoid(gate) * torch.tanh(filter_out)  # [B, D, T]
        
        # Apply FiLM with content condition
        if condition is not None:
            conv_out = self.film_condition(conv_out, condition)
       
        # Apply FiLM with time embedding
        conv_out = self.film_time(conv_out, time_emb)
        
        # Q-K-V attention with speaker prompt
        # [B, D, T] -> [B, T, D]
        conv_out_t = conv_out.transpose(1, 2)
       
        attn_out, _ = self.attention(
            query=conv_out_t,
            key=speaker_prompt,
            value=speaker_prompt
        )  # [B, T, D]
        
        # [B, T, D] -> [B, D, T]
        attn_out = attn_out.transpose(1, 2)
        
        # Output projection
        output = self.output_projection(attn_out)
        
        # Residual connection
        residual = self.residual_projection(residual)
        output = output + residual
        
        return output


class WaveNet(nn.Module):
    """
    WaveNet backbone for latent diffusion model.
    Contains M blocks with increasing dilation.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_blocks: int = 20,
        kernel_size: int = 3,
        dilation_rate: int = 2,
        condition_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # WaveNet blocks with increasing dilation
        self.blocks = nn.ModuleList([
            WaveNetBlock(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation_rate ** (i % 10),
                condition_dim=condition_dim,
                num_heads=num_heads
            )
            for i in range(num_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        )
    
    def get_time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        p_c: torch.Tensor,
        z_p: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_t: [B, D_latent, T] noisy latent
            t: [B] time step
            p_c: [B, T, D_cond] content condition
            z_p: [B, T_p, D_spk] speaker prompt
        
        Returns:
            z_0_pred: [B, D_latent, T] predicted clean latent
        """
        # Input projection
        x = self.input_projection(z_t)  # [B, D_hidden, T]
        
        # Time embedding
        time_emb = self.get_time_embedding(t, self.hidden_dim)
        time_emb = self.time_embedding(time_emb)  # [B, D_hidden]
        
        # Pass through WaveNet blocks
        for block in self.blocks:
            x = block(x, p_c, time_emb, z_p)
        
        # Output projection
        z_0_pred = self.output_projection(x)  # [B, D_latent, T]
        
        return z_0_pred
