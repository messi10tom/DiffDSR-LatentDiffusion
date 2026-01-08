from torch.utils.data import Dataset
from typing import Dict, List
from pathlib import Path

from typing import Optional
from .phoneme_utils import PhonemeConverter
from utils.audio import load_audio


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset for content encoder pretraining."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train-clean-100",
        sample_rate: int = 16000,
        limit: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.phoneme_converter = PhonemeConverter()

        self.data = self._load_metadata()
        self.limit = limit

    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata."""
        data = []
        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            return data

        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if trans_file.exists():
                    with open(trans_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                utt_id, text = parts
                                audio_path = chapter_dir / f"{utt_id}.flac"
                                if audio_path.exists():
                                    data.append({
                                        'audio_path': str(audio_path),
                                        'text': text,
                                        'speaker_id': speaker_dir.name
                                    })

        return data

    def __len__(self):
        return min(len(self.data), self.limit) if self.limit else len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        waveform = load_audio(item['audio_path'], self.sample_rate)

        # Placeholder phoneme sequence (would need G2P model in practice)
        phonemes = ['SIL'] + item['text'].split()[:10] + ['SIL']
        phoneme_ids = self.phoneme_converter.encode(phonemes)

        return {
            'waveform': waveform,
            'phoneme_ids': phoneme_ids,
            'speaker_id': item['speaker_id'],
            'text': item['text'],
        }


class UASpeechDataset(Dataset):
    """UASpeech dataset for dysarthric speech."""
    
    def __init__(
        self,
        root_dir: str,
        speaker_ids: List[str],
        sample_rate: int = 16000,
        limit: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.speaker_ids = speaker_ids
        self.sample_rate = sample_rate
        self.phoneme_converter = PhonemeConverter()
        
        self.data = self._load_metadata()
        self.limit = limit
    
    def _load_metadata(self) -> List[Dict]:
        """Load UASpeech metadata."""
        data = []
        
        for speaker_id in self.speaker_ids:
            speaker_dir = self.root_dir / speaker_id
            if not speaker_dir.exists():
                continue
            
            for audio_file in speaker_dir.glob("*.wav"):
                # Extract word from filename (UASpeech naming convention)
                word = audio_file.stem.split('_')[0] if '_' in audio_file.stem else audio_file.stem
                
                data.append({
                    'audio_path': str(audio_file),
                    'word': word,
                    'speaker_id': speaker_id
                })
        
        return data
    
    def __len__(self):
        return min(len(self.data), self.limit) if self.limit else len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        waveform = load_audio(item['audio_path'], self.sample_rate)
        
        # Placeholder phoneme sequence
        phonemes = ['SIL'] + list(item['word']) + ['SIL']
        phoneme_ids = self.phoneme_converter.encode(phonemes)
        
        return {
            'waveform': waveform,
            'phoneme_ids': phoneme_ids,
            'speaker_id': item['speaker_id'],
            'word': item['word'],
            'blank': self.phoneme_converter.phoneme_to_id['<BLANK>']
        }


class LibriTTSDataset(Dataset):
    """LibriTTS dataset for diffusion model training."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train-clean-100",
        sample_rate: int = 16000
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.phoneme_converter = PhonemeConverter()
        
        self.data = self._load_metadata()
    
    def _load_metadata(self) -> List[Dict]:
        """Load LibriTTS metadata."""
        data = []
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            return data
        
        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                for audio_file in chapter_dir.glob("*.wav"):
                    txt_file = audio_file.with_suffix('.normalized.txt')
                    if txt_file.exists():
                        with open(txt_file, 'r') as f:
                            text = f.read().strip()
                        data.append({
                            'audio_path': str(audio_file),
                            'text': text,
                            'speaker_id': speaker_dir.name
                        })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        waveform = load_audio(item['audio_path'], self.sample_rate)
        
        # Placeholder phoneme sequence
        phonemes = ['SIL'] + item['text'].split()[:30] + ['SIL']
        phoneme_ids = self.phoneme_converter.encode(phonemes)
        
        return {
            'waveform': waveform,
            'phoneme_ids': phoneme_ids,
            'speaker_id': item['speaker_id'],
            'text': item['text'],
            'blank': self.phoneme_converter.phoneme_to_id['<BLANK>']
        }


class VCTKDataset(Dataset):
    """VCTK dataset for speaker verification training."""
    
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000
    ):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        
        self.data = self._load_metadata()
    
    def _load_metadata(self) -> List[Dict]:
        """Load VCTK metadata."""
        data = []
        wav_dir = self.root_dir / "wav48"
        
        if not wav_dir.exists():
            return data
        
        for speaker_dir in wav_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            for audio_file in speaker_dir.glob("*.wav"):
                data.append({
                    'audio_path': str(audio_file),
                    'speaker_id': speaker_dir.name
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        waveform = load_audio(item['audio_path'], self.sample_rate)
        
        return {
            'waveform': waveform,
            'speaker_id': item['speaker_id']
        }
