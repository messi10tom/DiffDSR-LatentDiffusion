import torch
import numpy as np
import pyworld as pw
from typing import Optional


def extract_pitch(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 256,
    f0_min: float = 71.0,
    f0_max: float = 800.0
) -> torch.Tensor:
    """Extract F0 using PyWorld."""
    waveform = waveform.astype(np.float64)
    
    f0, timeaxis = pw.dio(
        waveform,
        sample_rate,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=hop_length / sample_rate * 1000
    )
    
    f0 = pw.stonemask(waveform, f0, timeaxis, sample_rate)
    
    # Convert to log scale, handle unvoiced frames
    f0_log = np.log(f0 + 1e-8)
    f0_log[f0 == 0] = 0
    
    return torch.from_numpy(f0_log).float()


def interpolate_pitch(pitch: torch.Tensor) -> torch.Tensor:
    """Interpolate unvoiced pitch values."""
    pitch = pitch.clone()
    voiced_mask = pitch > 0
    
    if voiced_mask.sum() == 0:
        return pitch
    
    indices = torch.arange(len(pitch))
    voiced_indices = indices[voiced_mask]
    voiced_values = pitch[voiced_mask]
    
    if len(voiced_indices) > 1:
        pitch = torch.from_numpy(
            np.interp(
                indices.numpy(),
                voiced_indices.numpy(),
                voiced_values.numpy()
            )
        ).float()
    
    return pitch
