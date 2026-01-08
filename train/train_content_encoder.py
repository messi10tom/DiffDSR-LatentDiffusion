import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import yaml

from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from models import SSLContentEncoder
from data import LibriSpeechDataset, UASpeechDataset
from utils import setup_logger


def train_content_encoder(config_path: str = "configs/default.yaml"):
    """Train content encoder on LibriSpeech then finetune on UASpeech."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('content_encoder', config['paths']['logs'])
    blank_id = 0
    
    # Create model
    model = SSLContentEncoder(
        ssl_type=config['model']['ssl_type'],
        ssl_model_name=config['model']['ssl_model_name'],
        n_phonemes=config['model']['n_phonemes'],
        conv_channels=config['model']['conv_channels'],
        conv_kernel_size=config['model']['conv_kernel_size'],
        freeze_ssl=True
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['training']['content_encoder_lr']),
        eps=float(config['training']['adam_eps'])
    )
    
    logger.info("Phase 1: Pretraining on LibriSpeech")
    
    train_dataset = LibriSpeechDataset(
        root_dir=config['paths']['librispeech'],
        split="train-clean-100",
        sample_rate=config['data']['sample_rate'],
        limit=20
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['content_encoder_batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    model.train()
    num_steps = config['training']['content_encoder_pretrain_steps']
    
    pbar = tqdm(total=num_steps, desc="Pretraining")
    step = 0
    
    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break
            
            waveform = batch['waveform'].to(device)
            phoneme_ids = batch['phoneme_ids'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            with autocast('cuda', dtype=torch.float16):           
                # Forward pass
                outputs = model(waveform)
                
                T = outputs["phoneme_logits"].size(0)
                input_lengths = torch.full(
                    (config['training']['content_encoder_batch_size'],),
                    T,
                    device=outputs['phoneme_logits'].device,
                    dtype=torch.long
                )
                # CTC loss
                _ , log_probs = model.compute_ctc_loss(
                    outputs['phoneme_logits'],
                    phoneme_ids,
                    input_lengths,
                    target_lengths
                )
                loss = F.ctc_loss(
                    log_probs,
                    phoneme_ids,
                    input_lengths,
                    target_lengths,
                    blank=blank_id,
                    zero_infinity=True
                )                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
                optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})
            
            step += 1
            
            if step % 10 == 0:
                checkpoint_path = os.path.join(
                    config['paths']['checkpoints'],
                    f'content_encoder_pretrain_step_{step}.pt'
                )
                if not os.path.exists(config['paths']['checkpoints']):
                    os.makedirs(config['paths']['checkpoints'])
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    pbar.close()
    
    logger.info("Phase 2: Finetuning on UASpeech")
    
    # Target speakers (lowest intelligibility)
    target_speakers = ["M12", "F02", "M16", "F04"]
    
    for speaker in target_speakers:
        logger.info(f"Finetuning for speaker: {speaker}")
        
        finetune_dataset = UASpeechDataset(
            root_dir=config['paths']['uaspeech'],
            speaker_ids=[speaker],
            sample_rate=config['data']['sample_rate'],
            limit=10
        )
        
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=config['training']['content_encoder_batch_size'],
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        model.train()
        num_finetune_steps = config['training']['content_encoder_finetune_steps']
        
        pbar = tqdm(total=num_finetune_steps, desc=f"Finetuning {speaker}")
        step = 0
        
        while step < num_finetune_steps:
            for batch in finetune_loader:
                if step >= num_finetune_steps:
                    break
                
                waveform = batch['waveform'].to(device)
                phoneme_ids = batch['phoneme_ids'].to(device)
                target_lengths = batch['target_lengths'].to(device)
                
                # Forward pass
                outputs = model(waveform)
                B = waveform.size(0)
                T = outputs["phoneme_logits"].size(0)
                input_lengths = torch.full(
                    (B,),
                    T,
                    device=outputs['phoneme_logits'].device,
                    dtype=torch.long
                )

                print("\n"*10, "="*50)
                print("Batch size:", B)
                print("T (max input length):", T)
                print("Input lengths:", input_lengths)
                print("="*50, "\n"*5)
                # CTC loss
                _ , log_probs = model.compute_ctc_loss(
                    outputs['phoneme_logits'],
                    phoneme_ids,
                    input_lengths,
                    target_lengths
                )
 
                loss = F.ctc_loss(
                    log_probs,
                    phoneme_ids,
                    input_lengths,
                    target_lengths,
                    blank=blank_id,
                    zero_infinity=True
                )               
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
                optimizer.step()
                
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
                
                step += 1
        
        pbar.close()
        
        # Save finetuned model
        checkpoint_path = os.path.join(
            config['paths']['checkpoints'],
            f'content_encoder_{speaker}.pt'
        )
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")


def collate_fn(batch):
    """Collate function for dataloader."""
    waveforms = [item['waveform'] for item in batch]
    phoneme_ids = [item['phoneme_ids'] for item in batch]
    
    # Pad waveforms
    max_wav_len = max([w.shape[0] for w in waveforms])
    padded_waveforms = torch.zeros(len(waveforms), max_wav_len)
   
    # Pad phoneme IDs
    max_phn_len = max([p.shape[0] for p in phoneme_ids])
    padded_phonemes = torch.zeros(len(phoneme_ids), max_phn_len, dtype=torch.long)
    target_lengths = torch.zeros(len(phoneme_ids), dtype=torch.long)
    
    for i, p in enumerate(phoneme_ids):
        padded_phonemes[i, :p.shape[0]] = p
        target_lengths[i] = p.shape[0]
    
    return {
        'waveform': padded_waveforms,
        'phoneme_ids': padded_phonemes,
        'target_lengths': target_lengths
    }


if __name__ == "__main__":
    train_content_encoder()
