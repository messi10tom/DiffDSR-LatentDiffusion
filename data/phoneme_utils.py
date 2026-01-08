import torch
from typing import List


class PhonemeConverter:
    """Convert between text and phoneme indices."""
    
    def __init__(self, phoneme_set: str = "arpabet"):
        self.phoneme_set = phoneme_set
        
        # ARPAbet phoneme set (simplified)
        self.phonemes = [
            "<BLANK>",
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN', 'PAD'
        ]
        
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.phonemes)}
        
    def __len__(self):
        return len(self.phonemes)
    
    def encode(self, phonemes: List[str]) -> torch.Tensor:
        """Convert phoneme list to indices."""
        indices = [self.phoneme_to_id.get(p, self.phoneme_to_id['SPN']) for p in phonemes]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> List[str]:
        """Convert indices to phoneme list."""
        if indices.dim() > 1:
            indices = indices.squeeze()
        return [self.id_to_phoneme[int(idx.item())] for idx in indices]
