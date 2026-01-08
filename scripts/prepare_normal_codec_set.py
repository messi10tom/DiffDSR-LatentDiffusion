import torch
import os
from tqdm import tqdm
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.codec import CodecModel
from data import LibriTTSDataset
from utils import setup_logger


def prepare_normal_codec_set(
    config_path: str = "configs/default.yaml",
    max_samples: int = 100,
):
    """
    Prepare normal codec set from VCTK dataset.
    This creates the codec set Z used in Equation (1) for codec normalization.
    
    Args:
        config_path: path to config file
        max_samples: maximum number of codec samples to extract
        samples_per_speaker: number of samples per speaker
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('prepare_codec_set', config['paths']['logs'])
    
    logger.info("Preparing normal codec set from VCTK dataset...")
    
    # Load codec model
    codec_model = CodecModel(
        sample_rate=config['data']['sample_rate']
    ).to(device)
    codec_model.eval()
    
    from data import LibriSpeechDataset
    dataset = LibriSpeechDataset(
        root_dir=config['paths']['librispeech'],
        split="train-clean-100",
        sample_rate=config['data']['sample_rate']
        )
    logger.info(f"Loaded LibriTTS dataset with {len(dataset)} samples")
    
    print("root_dir:", config['paths']['libritts'])
    # DataLoader
    def collate_fn(batch):
        waveforms = [item['waveform'] for item in batch]
        max_len = max([w.shape[0] for w in waveforms])
        
        # Pad waveforms
        padded = torch.zeros(len(waveforms), max_len)
        for i, w in enumerate(waveforms):
            padded[i, :w.shape[0]] = w
        
        return {'waveform': padded}
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Extract codec features
    all_codecs = []
    
    logger.info("Extracting codec features...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            waveform = batch['waveform'].to(device)
            try:
                # Encode to codec space
                codec_features = codec_model.encode(waveform)  # [B, D, T']
                
                # Transpose to [B, T', D]
                codec_features = codec_features.transpose(1, 2)
                
                print("codec_features shape:", codec_features.shape)
                # Move to CPU and store
                for i in range(codec_features.shape[0]):
                    codec = codec_features[i].cpu()
                    all_codecs.append(codec)
                    
                    if len(all_codecs) >= max_samples:
                        break
                
                if len(all_codecs) >= max_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to encode batch: {e}")
                continue
    
    logger.info(f"Extracted {len(all_codecs)} codec samples")
    
    # Stack into tensor
    # Pad to same length
    max_time = max([c.shape[0] for c in all_codecs])
    codec_dim = all_codecs[0].shape[1]
    
    normal_codec_set = torch.zeros(len(all_codecs), max_time, codec_dim)
    
    for i, codec in enumerate(all_codecs):
        t = codec.shape[0]
        normal_codec_set[i, :t, :] = codec
    
    logger.info(f"Normal codec set shape: {normal_codec_set.shape}")
    
    # Save to file
    os.makedirs(os.path.dirname(config['paths']['normal_codec_set']), exist_ok=True)
    torch.save(normal_codec_set, config['paths']['normal_codec_set'])
    
    logger.info(f"Saved normal codec set to: {config['paths']['normal_codec_set']}")
    logger.info("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare normal codec set for DSR training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=1000,
        help='Maximum number of codec samples'
    )
    parser.add_argument(
        '--samples_per_speaker',
        type=int,
        default=10,
        help='Samples per speaker'
    )
    
    args = parser.parse_args()
    
    prepare_normal_codec_set(
        config_path=args.config,
        max_samples=args.max_samples,
    )
