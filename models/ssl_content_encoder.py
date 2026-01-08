import torch
import torch.nn as nn
from typing import Tuple
from transformers import Wav2Vec2Model, HubertModel, WavLMModel
from typing import Dict, Optional


class SSLContentEncoder(nn.Module):
    """
    Speech content encoder using SSL models (Wav2Vec2, HuBERT, WavLM).
    Extracts phoneme embedding via CTC layer.
    """
    
    def __init__(
        self,
        ssl_type: str = "hubert",
        ssl_model_name: str = "facebook/hubert-base-ls960",
        n_phonemes: int = 42,
        conv_channels: list = [512, 512, 512],
        conv_kernel_size: int = 3,
        freeze_ssl: bool = True
    ):
        super().__init__()
        
        self.ssl_type = ssl_type
        self.freeze_ssl = freeze_ssl
        
        # Load SSL model
        if ssl_type == "wav2vec2":
            self.ssl_model = Wav2Vec2Model.from_pretrained(ssl_model_name)
        elif ssl_type == "hubert":
            self.ssl_model = HubertModel.from_pretrained(ssl_model_name)
        elif ssl_type == "wavlm":
            self.ssl_model = WavLMModel.from_pretrained(ssl_model_name)
        else:
            raise ValueError(f"Unknown SSL type: {ssl_type}")
        
        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        
        ssl_hidden_size = self.ssl_model.config.hidden_size
        
        # Convolution layers for local context
        conv_layers = []
        in_channels = ssl_hidden_size
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # CTC layer for phoneme prediction
        self.ctc_layer = nn.Linear(conv_channels[-1], n_phonemes)
        
    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: [B, T] audio waveform
            attention_mask: [B, T] attention mask
        
        Returns:
            phoneme_embedding: [B, T', n_phonemes] phoneme probability distribution
            hidden_features: [B, T', D] hidden features before CTC
        """
        # Extract SSL features
        if self.freeze_ssl:
            with torch.no_grad():
                ssl_outputs = self.ssl_model(
                    waveform,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
        else:
            ssl_outputs = self.ssl_model(
                waveform,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # [B, T', D]
        hidden_states = ssl_outputs.last_hidden_state
        
        # Apply convolution layers: [B, T', D] -> [B, D, T'] -> [B, D', T']
        hidden_states = hidden_states.transpose(1, 2)
        conv_features = self.conv_layers(hidden_states)
        conv_features = conv_features.transpose(1, 2)  # [B, T', D']
        
        # CTC prediction: [B, T', n_phonemes]
        phoneme_logits = self.ctc_layer(conv_features)
        phoneme_embedding = torch.log_softmax(phoneme_logits, dim=-1)
        
        return {
            'phoneme_embedding': phoneme_embedding,
            'phoneme_logits': phoneme_logits,
            'hidden_features': conv_features
        }
    
    def compute_ctc_loss(
        self,
        phoneme_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CTC loss.
        
        Args:
            phoneme_logits: [B, T, n_phonemes]
            targets: [B, S] target phoneme indices
            input_lengths: [B] input sequence lengths
            target_lengths: [B] target sequence lengths
        """
        log_probs = torch.log_softmax(phoneme_logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # [T, B, n_phonemes]
        
        loss = nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction='mean',
            zero_infinity=True
        )
        
        return loss, log_probs
