import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.duration import expand_with_duration
from models.speaker_identity_encoder import InContextAttention

class DurationPredictor(nn.Module):
    """Duration predictor with convolution layers."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: list = [256, 256],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.LayerNorm(out_channels),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_channels[-1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input phoneme embeddings
        
        Returns:
            duration: [B, N] predicted duration
        """
        # [B, N, D] -> [B, D, N]
        x = x.transpose(1, 2)
        
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x)
            elif isinstance(layer, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = layer(x)
                x = x.transpose(1, 2)
            else:
                x = layer(x)
        
        # [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)
        
        # [B, N, 1] -> [B, N]
        duration = self.output_layer(x).squeeze(-1)
        
        # Ensure positive durations with minimum value
        # Use softplus instead of exp for stability
        duration = torch.nn.functional.softplus(duration) + 0.5
        duration = torch.clamp(duration, min=1.0, max=100.0)  # Reasonable bounds
        
        return duration


class PitchPredictor(nn.Module):
    """Pitch predictor with convolution layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_channels: list = [256, 256],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        in_channels = input_dim

        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.LayerNorm(out_channels),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input phoneme embeddings

        Returns:
            pitch: [B, N] predicted pitch
        """
        # [B, N, D] -> [B, D, N]
        x = x.transpose(1, 2)
        
        print("="*50)
        print("PitchPredictor input x:", x.shape)
        print("="*50)
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x)
            elif isinstance(layer, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = layer(x)
                x = x.transpose(1, 2)
            else:
                x = layer(x)

        # [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)

        # [B, N, 1] -> [B, N]
        pitch = self.output_layer(x).squeeze(-1)

        return pitch


class VarianceAdaptor(nn.Module):
    """
    Variance adaptor with duration and pitch predictors.
    Uses in-context learning attention with speaker prompt.
    """

    def __init__(
        self,
        input_dim: int,
        duration_predictor_channels: list = [256, 256],
        pitch_predictor_channels: list = [256, 256],
        variance_kernel_size: int = 3,
        pitch_embedding_dim: int = 256
    ):
        super().__init__()

        # Duration predictor
        self.duration_predictor = DurationPredictor(
            input_dim=input_dim,
            hidden_channels=duration_predictor_channels,
            kernel_size=variance_kernel_size
        )

        # Pitch predictor
        self.pitch_predictor = PitchPredictor(
            input_dim=input_dim,
            hidden_channels=pitch_predictor_channels,
            kernel_size=variance_kernel_size
        )

        # In-context attention for duration predictor
        self.duration_attention = InContextAttention(
            hidden_dim=input_dim,
            num_heads=8
        )

        # In-context attention for pitch predictor
        self.pitch_attention = InContextAttention(
            hidden_dim=input_dim,
            num_heads=8
        )

        # Pitch embedding
        self.pitch_embedding = nn.Linear(1, pitch_embedding_dim)

        # Output projection
        self.output_projection = nn.Linear(input_dim + pitch_embedding_dim, input_dim)


    def forward(
        self,
        phoneme_embedding: torch.Tensor,
        speaker_prompt: torch.Tensor,
        target_duration: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            phoneme_embedding: [B, N, D] phoneme embeddings
            speaker_prompt: [B, T_p, D] speaker-aware representation
            target_duration: [B, N] ground truth duration (optional)
            target_pitch: [B, T] ground truth pitch (optional)
        
        Returns:
            p_c: [B, T, D] expanded condition with duration and pitch
            predicted_duration: [B, N]
            predicted_pitch: [B, T]
        """
        # Apply in-context attention for duration
        duration_query = self.duration_attention(
            query=phoneme_embedding,
            key=speaker_prompt,
            value=speaker_prompt
        )
        
        # Predict duration
        predicted_duration = self.duration_predictor(duration_query)  # [B, N]
        
        # Use target or predicted duration
        if target_duration is not None:
            duration = target_duration
            duration = torch.clamp(duration, min=1)  # Ensure minimum duration
        else:
            duration = predicted_duration.round().long()
            duration = torch.clamp(duration, min=1)  # Ensure minimum duration
        
        # Expand phoneme embedding with duration
        expanded_embedding = expand_with_duration(phoneme_embedding, duration)  # [B, T, D]
        
        # Safety check for empty expansion
        if expanded_embedding.shape[1] == 0:
            # Fallback: use identity expansion
            expanded_embedding = phoneme_embedding
            duration = torch.ones_like(duration)
        
        # Apply in-context attention for pitch
        pitch_query = self.pitch_attention(
            query=expanded_embedding,
            key=speaker_prompt,
            value=speaker_prompt
        )
        
        # Predict pitch
        predicted_pitch = self.pitch_predictor(pitch_query)  # [B, T]
        
        # Use target or predicted pitch
        if target_pitch is not None:
            # Match pitch length to expanded embedding
            if target_pitch.shape[1] != expanded_embedding.shape[1]:
                target_pitch = torch.nn.functional.interpolate(
                    target_pitch.unsqueeze(1),
                    size=expanded_embedding.shape[1],
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
            pitch = target_pitch
        else:
            pitch = predicted_pitch
        
        # Embed pitch
        pitch_emb = self.pitch_embedding(pitch.unsqueeze(-1))  # [B, T, D_pitch]
        
        # Combine expanded embedding and pitch
        combined = torch.cat([expanded_embedding, pitch_emb], dim=-1)  # [B, T, D + D_pitch]
        
        # Project to final condition
        p_c = self.output_projection(combined)  # [B, T, D]
        
        return {
            'p_c': p_c,
            'predicted_duration': predicted_duration,
            'predicted_pitch': predicted_pitch
        }
