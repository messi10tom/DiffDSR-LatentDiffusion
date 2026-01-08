import torch
import torchaudio
from typing import Optional


def load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int = 16000):
    """Save audio waveform to file."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu(), sample_rate)


def mel_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    sample_rate: int = 16000,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0
) -> torch.Tensor:
    """Compute mel spectrogram."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax
    )
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel
