import torch
from jiwer import wer
from typing import List


def compute_per(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Compute Phoneme Error Rate (PER).
    
    Args:
        predictions: list of predicted phoneme sequences (space-separated)
        references: list of reference phoneme sequences (space-separated)
    
    Returns:
        per: phoneme error rate
    """
    # WER at phoneme level is equivalent to PER
    per = wer(references, predictions)
    return per


def decode_phonemes(phoneme_logits: torch.Tensor, phoneme_converter) -> List[str]:
    """
    Decode phoneme logits to phoneme sequences.
    
    Args:
        phoneme_logits: [B, T, n_phonemes] logit scores
        phoneme_converter: PhonemeConverter instance
    
    Returns:
        phoneme_sequences: list of phoneme strings
    """
    # Greedy decoding
    pred_indices = phoneme_logits.argmax(dim=-1)  # [B, T]
    
    phoneme_sequences = []
    for i in range(pred_indices.shape[0]):
        indices = pred_indices[i]
        
        # Remove consecutive duplicates and blanks
        collapsed_indices = []
        prev_idx = -1
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != 0:  # 0 is blank
                collapsed_indices.append(idx)
            prev_idx = idx
        
        # Convert to phonemes
        phonemes = phoneme_converter.decode(torch.tensor(collapsed_indices))
        phoneme_sequences.append(' '.join(phonemes))
    
    return phoneme_sequences
