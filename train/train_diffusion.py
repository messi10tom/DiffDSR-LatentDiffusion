import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import yaml

from models import (
    SSLContentEncoder,
    SpeakerIdentityEncoder,
    VarianceAdaptor,
    DiffusionGenerator,
    CodecModel
)
from data import LibriTTSDataset
from utils import setup_logger


def train_diffusion(config_path: str = "configs/default.yaml"):
    """Train diffusion generator on LibriTTS."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('diffusion', config['paths']['logs'])
    
    # Create models
    content_encoder = SSLContentEncoder(
        ssl_type=config['model']['ssl_type'],
        ssl_model_name=config['model']['ssl_model_name'],
        n_phonemes=config['model']['n_phonemes'],
        conv_channels=config['model']['conv_channels'],
        freeze_ssl=True
    ).to(device)
    
    # Load pretrained content encoder
    content_encoder_path = os.path.join(
        config['paths']['checkpoints'],
        'content_encoder_pretrain_step_1000000.pt'
    )
    if os.path.exists(content_encoder_path):
        content_encoder.load_state_dict(torch.load(content_encoder_path))
        logger.info(f"Loaded content encoder from {content_encoder_path}")
    
    content_encoder.eval()
    for param in content_encoder.parameters():
        param.requires_grad = False
    
    speaker_encoder = SpeakerIdentityEncoder(
        codec_dim=config['model']['codec_dim'],
        speaker_hidden_dim=config['model']['speaker_hidden_dim'],
        speaker_transformer_layers=config['model']['speaker_transformer_layers'],
        speaker_nhead=config['model']['speaker_nhead'],
        normal_codec_set_path=config['paths']['normal_codec_set']
    ).to(device)
    
    variance_adaptor = VarianceAdaptor(
        input_dim=config['model']['n_phonemes'],
        duration_predictor_channels=config['model']['duration_predictor_channels'],
        pitch_predictor_channels=config['model']['pitch_predictor_channels'],
        variance_kernel_size=config['model']['variance_kernel_size'],
        pitch_embedding_dim=256
    ).to(device)
    
    diffusion_model = DiffusionGenerator(
        latent_dim=config['model']['codec_dim'],
        wavenet_channels=config['model']['wavenet_channels'],
        wavenet_blocks=config['model']['wavenet_blocks'],
        wavenet_kernel_size=config['model']['wavenet_kernel_size'],
        wavenet_dilation_rate=config['model']['wavenet_dilation_rate'],
        condition_dim=config['model']['n_phonemes'],
        speaker_dim=config['model']['speaker_hidden_dim'],
        beta_min=config['model']['beta_min'],
        beta_max=config['model']['beta_max'],
        diffusion_steps=config['model']['diffusion_steps']
    ).to(device)
    
    codec_model = CodecModel(
        sample_rate=config['data']['sample_rate']
    ).to(device)
    
    # Optimizer (only train variance adaptor and diffusion model)
    trainable_params = list(variance_adaptor.parameters()) + \
                      list(diffusion_model.parameters()) + \
                      list(speaker_encoder.parameters())
    
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(config['training']['diffusion_lr']),
        betas=(float(config['training']['adam_beta1']), float(config['training']['adam_beta2'])),
        eps=float(config['training']['adam_eps'])
    )
    
    # Dataset
    from data import LibriSpeechDataset
    train_dataset = LibriSpeechDataset(
        root_dir=config['paths']['librispeech'],
        split="train-clean-100",
        sample_rate=config['data']['sample_rate']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['diffusion_batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_diffusion
    )
    
    # Training loop
    num_steps = config['training']['diffusion_steps_total']
    
    pbar = tqdm(total=num_steps, desc="Training Diffusion")
    step = 0
    
    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break
            
            waveform = batch['waveform'].to(device)
            
            # Extract phoneme embedding (frozen)
            with torch.no_grad():
                content_outputs = content_encoder(waveform)
                phoneme_embedding = content_outputs['phoneme_embedding']  # [B, T', n_phn]
            
            # Extract speaker prompt
            speaker_prompt = speaker_encoder(waveform, apply_normalization=True)  # [B, T_p, D]
            
            # Variance adaptor (with dummy duration/pitch for teacher forcing)
            # In practice, would extract from alignment
            variance_outputs = variance_adaptor(
                phoneme_embedding=phoneme_embedding,
                speaker_prompt=speaker_prompt,
                target_duration=None,
                target_pitch=None
            )
            
            p_c = variance_outputs['p_c']  # [B, T, D]
            
            # Extract ground truth codec latents
            with torch.no_grad():
                z_0 = codec_model.encode(waveform)  # [B, D, T]
            
            # Compute diffusion loss
            loss = diffusion_model.compute_loss(z_0, p_c, speaker_prompt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                config['training']['gradient_clip']
            )
            optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})
            
            step += 1
            
            if step % 10000 == 0:
                # Save checkpoint
                checkpoint = {
                    'step': step,
                    'speaker_encoder': speaker_encoder.state_dict(),
                    'variance_adaptor': variance_adaptor.state_dict(),
                    'diffusion_model': diffusion_model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                checkpoint_path = os.path.join(
                    config['paths']['checkpoints'],
                    f'diffusion_step_{step}.pt'
                )
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    pbar.close()
    logger.info("Training complete!")


def collate_fn_diffusion(batch):
    """Collate function for diffusion training."""
    waveforms = [item['waveform'] for item in batch]
    
    # Pad waveforms
    max_len = max([w.shape[0] for w in waveforms])
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w
    
    return {
        'waveform': padded_waveforms
    }


if __name__ == "__main__":
    train_diffusion()
