import torch

def extract_duration(alignment: torch.Tensor) -> torch.Tensor:
    """
    Extract duration from alignment matrix.
    
    Args:
        alignment: [T, N] alignment matrix
    
    Returns:
        duration: [N] duration for each phoneme
    """
    duration = alignment.sum(dim=0)
    return duration.long()


def expand_with_duration(x: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
    """
    Expand sequence according to duration.
    
    Args:
        x: [B, N, D] input sequence
        duration: [B, N] duration for each frame
    
    Returns:
        expanded: [B, T, D] expanded sequence
    """
    batch_size, seq_len, dim = x.shape
    
    # Ensure durations are positive integers and at least 1
    duration = duration.round().long()
    duration = torch.clamp(duration, min=1)  # Minimum duration of 1
    
    max_len = int(duration.sum(dim=1).max().item())
    
    # Safety check
    if max_len == 0:
        max_len = seq_len  # Fallback to input length
        duration = torch.ones_like(duration)
    
    expanded = torch.zeros(batch_size, max_len, dim, device=x.device, dtype=x.dtype)
    
    for b in range(batch_size):
        pos = 0
        for i in range(seq_len):
            dur = int(duration[b, i].item())
            if dur > 0 and pos < max_len:
                end_pos = min(pos + dur, max_len)
                expanded[b, pos:end_pos] = x[b, i:i+1].repeat(end_pos - pos, 1)
                pos = end_pos
    
    return expanded
