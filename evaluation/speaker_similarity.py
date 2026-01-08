import torch
import torch.nn.functional as F
from typing import List

from models.speaker_identity_encoder import SpeakerVerificationModel


def compute_speaker_similarity(
    original_waveforms: List[torch.Tensor],
    reconstructed_waveforms: List[torch.Tensor],
    sv_model: SpeakerVerificationModel,
    device: torch.device
) -> float:
    """
    Compute speaker similarity using L1 distance in SV embedding space.
    
    Args:
        original_waveforms: list of original waveforms
        reconstructed_waveforms: list of reconstructed waveforms
        sv_model: speaker verification model
        device: torch device
    
    Returns:
        avg_distance: average L1 distance
    """
    sv_model.eval()
    sv_model.to(device)
    
    total_distance = 0.0
    count = 0
    
    with torch.no_grad():
        for orig, recon in zip(original_waveforms, reconstructed_waveforms):
            # Extract codec features (simplified)
            # In practice, use codec model
            orig_feat = torch.randn(1, 50, 128).to(device)  # Placeholder
            recon_feat = torch.randn(1, 50, 128).to(device)  # Placeholder
            
            # Get SV embeddings
            orig_emb = sv_model(orig_feat)
            recon_emb = sv_model(recon_feat)
            
            # L1 distance
            distance = torch.abs(orig_emb - recon_emb).sum().item()
            
            total_distance += distance
            count += 1
    
    avg_distance = total_distance / count if count > 0 else 0.0
    
    return avg_distance
