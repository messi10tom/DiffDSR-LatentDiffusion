import torch
import torch.nn as nn

class CodecModel(nn.Module):
    """
    Wrapper for EnCodec model.
    Provides encoding and decoding functionality using pre-trained EnCodec.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        bandwidth: float = 6.0,
        use_continuous_latent: bool = True
    ):
        super().__init__()
        
        self.use_continuous_latent = use_continuous_latent
        
        try:
            # Try importing encodec
            from encodec.model import EncodecModel as _EncodecModel
            
            # Load pre-trained EnCodec model
            # EnCodec has 24kHz model which is the standard one
            self.codec = _EncodecModel.encodec_model_24khz()
            self.codec.set_target_bandwidth(bandwidth)
            
            self.sample_rate = sample_rate
            self.codec_sample_rate = 24000
            
            # Get the latent dimension from encoder output
            # EnCodec encoder outputs continuous embeddings before quantization
            self.latent_dim = 128  # Standard EnCodec dimension
            
            # Freeze codec parameters
            for param in self.codec.parameters():
                param.requires_grad = False
            
            self.encodec_available = True
            
        except ImportError:
            print("WARNING: encodec not installed. Using placeholder codec model.")
            print("Install with: pip install encodec")
            self.encodec_available = False
            self.sample_rate = sample_rate
            self.codec_sample_rate = 24000
            self.latent_dim = 128
            
            # Placeholder codec for when encodec is not available
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, stride=4, padding=3),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(128, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 64, kernel_size=7, stride=4, padding=3, output_padding=3),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 1, kernel_size=7, stride=4, padding=3, output_padding=3),
                nn.Tanh()
            )
    
    def _resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample waveform to target sample rate."""
        if orig_sr == target_sr:
            return waveform
        
        # Use torch interpolation for resampling
        scale_factor = target_sr / orig_sr
        resampled = torch.nn.functional.interpolate(
            waveform,
            scale_factor=scale_factor,
            mode='linear',
            align_corners=False
        )
        return resampled
    
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform to codec latents.
        
        Args:
            waveform: [B, T] or [B, 1, T]
        
        Returns:
            latents: [B, D, T'] codec latents where D=128 (continuous embeddings)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]
        
        # Resample to codec sample rate
        if self.sample_rate != self.codec_sample_rate:
            waveform = self._resample(waveform, self.sample_rate, self.codec_sample_rate)
        
        if self.encodec_available:
            with torch.no_grad():
                if self.use_continuous_latent:
                    # Get continuous embeddings from encoder (before quantization)
                    # This is the actual latent representation used in diffusion models
                    latents = self.codec.encoder(waveform)  # [B, 128, T']
                    
                else:
                    # Alternative: Use quantized codes
                    # EnCodec expects [B, C, T] format
                    encoded_frames = self.codec.encode(waveform)
                    
                    # EnCodec returns list of tuples (codes, scale)
                    # codes shape: [B, n_q, T'] where n_q is number of quantizers
                    codes = encoded_frames[0][0]  # Get codes from first frame
                    
                    # Use only first quantizer level for simplicity
                    # or average across quantizers
                    latents = codes[:, :1, :].float()  # [B, 1, T']
                    
                    # Expand to match expected dimension
                    if latents.shape[1] < self.latent_dim:
                        latents = latents.repeat(1, self.latent_dim, 1)
                
        else:
            # Placeholder encoding
            latents = self.encoder(waveform)  # [B, 128, T']
        
        return latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode codec latents to waveform.
        
        Args:
            latents: [B, D, T'] codec latents where D=128
        
        Returns:
            waveform: [B, 1, T] reconstructed waveform
        """
        if self.encodec_available:
            with torch.no_grad():
                if self.use_continuous_latent:
                    # Decode from continuous latents directly
                    # Pass through quantizer and decoder
                    qr = self.codec.quantizer(
                        latents, 
                        self.codec.frame_rate
                    )
                    quantized_latents = qr.quantized
                    waveform = self.codec.decoder(quantized_latents)
                    
                else:
                    # Decode from codes
                    # This path is more complex and requires proper code handling
                    # For simplicity, we'll just use the encoder->decoder path
                    quantized_latents, _, _ = self.codec.quantizer(
                        latents,
                        self.codec.frame_rate
                    )
                    waveform = self.codec.decoder(quantized_latents)
        else:
            # Placeholder decoding
            waveform = self.decoder(latents)  # [B, 1, T]
        
        # Resample back to target sample rate
        if self.sample_rate != self.codec_sample_rate:
            waveform = self._resample(waveform, self.codec_sample_rate, self.sample_rate)
        
        return waveform
    
    def encode_decode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode then decode (for testing).
        
        Args:
            waveform: [B, T] or [B, 1, T]
        
        Returns:
            reconstructed: [B, 1, T]
        """
        latents = self.encode(waveform)
        reconstructed = self.decode(latents)
        return reconstructed


def get_pretrained_codec(
    sample_rate: int = 16000, 
    bandwidth: float = 6.0,
    use_continuous_latent: bool = True
) -> CodecModel:
    """
    Get pre-trained codec model.
    
    Args:
        sample_rate: target sample rate
        bandwidth: target bandwidth for EnCodec
        use_continuous_latent: if True, use continuous embeddings; else use quantized codes
    
    Returns:
        codec_model: CodecModel instance
    """
    return CodecModel(
        sample_rate=sample_rate, 
        bandwidth=bandwidth,
        use_continuous_latent=use_continuous_latent
    )
