import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import CodecModel


class SpeakerVerificationModel(nn.Module):
    """Simplified speaker verification model for codec normalization."""
    
    def __init__(self, input_dim: int = 128, embedding_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] codec features
        
        Returns:
            embedding: [B, D'] speaker embedding
        """
        # Mean pooling
        x = x.mean(dim=1)
        embedding = self.encoder(x)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


class CodecNormalizer(nn.Module):
    """
    Codec normalizer using Eq. (1) from paper.
    Maps dysarthric codecs to normal codecs via L1 distance in SV space.
    """
    
    def __init__(
        self,
        normal_codec_set_path: str,
        sv_model: SpeakerVerificationModel,
        codec_dim: int = 128
    ):
        super().__init__()
        
        self.sv_model = sv_model
        self.codec_dim = codec_dim
        
        # Load normal codec set Z
        # Expected format: [N, T, D] tensor of normal speech codecs
        if normal_codec_set_path:
            try:
                self.normal_codec_set = torch.load(normal_codec_set_path)
            except:
                raise FileNotFoundError(f"Normal codec set file not found: {normal_codec_set_path}")
        else:
            raise ValueError("normal_codec_set_path must be provided.")
    
    def forward(self, dysarthric_codec: torch.Tensor) -> torch.Tensor:
        """
        Normalize dysarthric codec to normal codec.
        Eq. (1): ẑ_p → z̃_p = argmin |f_SV(ẑ_p) - f_SV(z̃_p)|
        
        Args:
            dysarthric_codec: [B, T, D] dysarthric codec
        
        Returns:
            normal_codec: [B, T, D] normalized normal codec
        """
        batch_size = dysarthric_codec.shape[0]
        device = dysarthric_codec.device
        
        # Move normal codec set to device
        normal_codec_set = self.normal_codec_set.to(device)
        
        # Compute SV embedding for dysarthric codec
        with torch.no_grad():
            dysarthric_embedding = self.sv_model(dysarthric_codec)  # [B, D']
        
        # Compute SV embeddings for all normal codecs
        num_normal = normal_codec_set.shape[0]
        normal_embeddings = []
        
        with torch.no_grad():
            for i in range(num_normal):
                normal_codec = normal_codec_set[i:i+1].repeat(batch_size, 1, 1)
                normal_emb = self.sv_model(normal_codec)
                normal_embeddings.append(normal_emb)
        
        normal_embeddings = torch.stack(normal_embeddings, dim=1)  # [B, N, D']
        
        # Compute L1 distance
        dysarthric_embedding = dysarthric_embedding.unsqueeze(1)  # [B, 1, D']
        distances = torch.abs(dysarthric_embedding - normal_embeddings).sum(dim=-1)  # [B, N]
        
        # Find nearest normal codec
        min_indices = distances.argmin(dim=1)  # [B]
        
        # Select normalized codecs
        normalized_codecs = []
        for b in range(batch_size):
            idx = min_indices[b].item()
            normalized_codecs.append(normal_codec_set[idx])
        
        normalized_codecs = torch.stack(normalized_codecs, dim=0)  # [B, T, D]
        
        return normalized_codecs


class InContextAttention(nn.Module):
    """Q-K-V attention for in-context learning."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_v, D]
        
        Returns:
            output: [B, T_q, D]
        """
        output, _ = self.attention(query, key, value)
        return output


class SpeakerIdentityEncoder(nn.Module):
    """
    Speaker identity encoder with in-context learning.
    Components:
    1. Pre-trained SE model (placeholder)
    2. Codec tokenizer (EnCodec)
    3. Codec normalizer
    4. Transformer layers for in-context learning
    """
    
    def __init__(
        self,
        normal_codec_set_path: str,
        codec_dim: int = 128,
        speaker_hidden_dim: int = 512,
        speaker_transformer_layers: int = 4,
        speaker_nhead: int = 8
    ):
        super().__init__()
        
        # Codec model
        self.codec_model = CodecModel()
        
        # Speaker verification model for normalization
        self.sv_model = SpeakerVerificationModel(
            input_dim=codec_dim,
            embedding_dim=256
        )
        
        # Codec normalizer
        self.codec_normalizer = CodecNormalizer(
            normal_codec_set_path=normal_codec_set_path,
            sv_model=self.sv_model,
            codec_dim=codec_dim
        )
        
        # Project codec to hidden dim
        self.codec_projection = nn.Linear(codec_dim, speaker_hidden_dim)
        
        # Transformer layers for in-context learning
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=speaker_hidden_dim,
            nhead=speaker_nhead,
            dim_feedforward=speaker_hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=speaker_transformer_layers
        )
        
    def forward(
        self,
        waveform: torch.Tensor,
        apply_normalization: bool = True
    ) -> torch.Tensor:
        """
        Extract speaker-aware representation.
        
        Args:
            waveform: [B, T] audio waveform
            apply_normalization: whether to apply codec normalization
        
        Returns:
            z_p: [B, T', D] speaker-aware representation
        """
        # Step 1: SE model (placeholder - just pass through)
        enhanced_waveform = waveform
        
        # Step 2: Codec encoding
        codec_features = self.codec_model.encode(enhanced_waveform)  # [B, D, T']
        codec_features = codec_features.transpose(1, 2)  # [B, T', D]
        
        # Step 3: Codec normalization
        if apply_normalization:
            codec_features = self.codec_normalizer(codec_features)
        
        # Step 4: Project to hidden dim
        speaker_features = self.codec_projection(codec_features)  # [B, T', D_hidden]
        
        # Step 5: Transformer for in-context learning
        z_p = self.transformer(speaker_features)  # [B, T', D_hidden]
        
        return z_p
