import torch
import yaml
import os

from typing import Optional
from models import (
    SSLContentEncoder,
    SpeakerIdentityEncoder,
    VarianceAdaptor,
    DiffusionGenerator,
    CodecModel
)
from utils import load_audio, save_audio


def inference(
    input_audio_path: str,
    output_audio_path: str,
    config_path: str = "configs/default.yaml",
    checkpoint_path: Optional[str] = None
):
    """
    Run inference on dysarthric speech.
    
    Args:
        input_audio_path: path to input dysarthric speech
        output_audio_path: path to save reconstructed speech
        config_path: path to config file
        checkpoint_path: path to trained checkpoint
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    content_encoder = SSLContentEncoder(
        ssl_type=config['model']['ssl_type'],
        ssl_model_name=config['model']['ssl_model_name'],
        n_phonemes=config['model']['n_phonemes'],
        conv_channels=config['model']['conv_channels'],
        freeze_ssl=True
    ).to(device)
    
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
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        speaker_encoder.load_state_dict(checkpoint['speaker_encoder'])
        variance_adaptor.load_state_dict(checkpoint['variance_adaptor'])
        diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Set to eval mode
    content_encoder.eval()
    speaker_encoder.eval()
    variance_adaptor.eval()
    diffusion_model.eval()
    codec_model.eval()
    
    # Load input audio
    waveform = load_audio(input_audio_path, config['data']['sample_rate'])
    waveform = waveform.unsqueeze(0).to(device)  # [1, T]
    
    # Inference
    with torch.no_grad():
        content_outputs = content_encoder(waveform)
        phoneme_embedding = content_outputs['phoneme_embedding']  # [1, T', n_phn]
        
        speaker_prompt = speaker_encoder(waveform, apply_normalization=True)  # [1, T_p, D]
        
        variance_outputs = variance_adaptor(
            phoneme_embedding=phoneme_embedding,
            speaker_prompt=speaker_prompt,
            target_duration=None,
            target_pitch=None
 
        )
        
        p_c = variance_outputs['p_c']  # [1, T, D]
        
        z_0 = diffusion_model.sample(
                p_c=p_c,
                z_p=speaker_prompt,
                num_steps=config['inference']['diffusion_sampling_steps'],
                temperature=config['inference']['temperature']
        )  # [1, D, T]
        
        reconstructed_waveform = codec_model.decode(z_0)  # [1, 1, T]
    
    # Save output
    save_audio(
        output_audio_path,
        reconstructed_waveform.squeeze(),
        config['data']['sample_rate']
    )
    
    print(f"Reconstructed speech saved to {output_audio_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input audio path')
    parser.add_argument('--output', type=str, required=True, help='Output audio path')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    
    inference(args.input, args.output, args.config, args.checkpoint)
