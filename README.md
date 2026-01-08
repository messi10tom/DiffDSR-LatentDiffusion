# DiffDSR: Dysarthric Speech Reconstruction Using Latent Diffusion Model

Unofficial implementation of "DiffDSR: Dysarthric Speech Reconstruction Using Latent Diffusion Model" (arXiv:2506.00350v1).

## Overview

DiffDSR is a diffusion-based dysarthric speech reconstruction system that converts dysarthric speech into comprehensible speech while maintaining speaker identity. The system leverages:

- **SSL Content Encoder**: Uses pre-trained self-supervised learning models (HuBERT/Wav2Vec2/WavLM) for robust phoneme extraction
- **Speaker Identity Encoder**: Preserves speaker-aware identity through codec normalization and in-context learning
- **Latent Diffusion Model**: Reconstructs speech using SDE-based diffusion with WaveNet backbone

## Architecture

```
Dysarthric Speech
	↓ 
[SSL Content Encoder] → Phoneme Embedding (p)
	↓
[Variance Adaptor] → Content Condition (p_c)
    ↓
[Speaker Identity Encoder] → Speaker Prompt (z_p)
    ↓
[Latent Diffusion Model] → Codec Latent (z_0)
    ↓
[Codec Decoder] → Reconstructed Speech
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/diffdsr.git
cd diffdsr

# Vitual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Datasets Required

1. **LibriSpeech**: For content encoder pretraining
2. **noise-reduced derivative of UASpeech**: For dysarthric speech finetuning

### Directory Structure

```
data/
├── librispeech/
│   └── train-clean-100/
├── uaspeech/
│   ├── M12/
│   ├── F02/
│   ├── M16/
│   └── F04/
├── vctk/
│   └── wav48/
└── libritts/
    └── train-clean-100/
```

## Download Dataset

```bash
mkdir -p data && cd data

# -------- LibriSpeech (content encoder) --------
mkdir -p librispeech
cd librispeech
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
mv LibriSpeech/train-clean-100 .
rm -rf LibriSpeech train-clean-100.tar.gz
cd ..

# ===== UASpeech (noise-reduced, Kaggle) setup =====
kaggle datasets download -d aryashah2k/noise-reduced-uaspeech-dysarthria-dataset

# Extract
unzip noise-reduced-uaspeech-dysarthria-dataset.zip -d uaspeech_raw
rm noise-reduced-uaspeech-dysarthria-dataset.zip

# Normalize directory structure
mkdir -p uaspeech
mv uaspeech_raw/noisereduced-uaspeech/* uaspeech/
rm -rf uaspeech_raw
cd ..

```
## Training

### Complete Training Pipeline

The training should be done in this order:

#### Step 0: Prepare Normal Codec Set (Required)
```bash
# This creates the normal codec set Z for codec normalization (Eq. 1)
python scripts/prepare_normal_codec_set.py --max_samples 1000
```

This will:
- Extract codec features from VCTK or LibriTTS (normal speech)
- Save to `./data/normal_codec_set.pt`
- Used for codec normalization in speaker identity encoder

#### Step 1: Train Content Encoder
```bash
python train/train_content_encoder.py
```

This will:
- Pretrain on LibriSpeech for 1M steps
- Finetune on UASpeech speakers (M12, F02, M16, F04) for 2K steps each
- Save checkpoints to `./checkpoints/content_encoder_*.pt`

#### Step 2: Train Diffusion Model
```bash
python train/train_diffusion.py
```

This will:
- Train on LibriTTS for 300K iterations
- Use pretrained content encoder (frozen)
- Save checkpoints every 10K steps to `./checkpoints/diffusion_step_*.pt`

### Quick Start (Minimal Setup)

If you don't have all datasets, you can still test the pipeline:
```bash
# 1. Create dummy normal codec set (for testing only)
mkdir -p data
python -c "import torch; torch.save(torch.randn(100, 50, 128), 'data/normal_codec_set.pt')"

# 2. Skip content encoder pretraining and go straight to diffusion
# (content encoder will initialize with random weights)
python train/train_diffusion.py
```

### Training Order Summary
```
1. prepare_normal_codec_set.py  ← Creates ./data/normal_codec_set.pt
   ↓
2. train_content_encoder.py     ← Uses LibriSpeech + UASpeech
   ↓
3. train_diffusion.py           ← Uses LibriTTS + normal_codec_set.pt
   ↓
4. inference.py                 ← Uses all trained models
```

## Inference

```bash
python train/inference.py \
    --input path/to/dysarthric_speech.wav \
    --output path/to/reconstructed_speech.wav \
    --checkpoint checkpoints/diffusion_step_300000.pt
```

## Evaluation

### Phoneme Error Rate (PER)

```python
from evaluation import compute_per

per = compute_per(predictions, references)
print(f"PER: {per:.2%}")
```

### Speaker Similarity

```python
from evaluation import compute_speaker_similarity

similarity = compute_speaker_similarity(
    original_waveforms,
    reconstructed_waveforms,
    sv_model,
    device
)
print(f"L1 Distance: {similarity:.4f}")
```

### Human Listening Test (MOS)

```python
from evaluation import mos_evaluation

mos_evaluation(audio_paths, save_path="mos_results.txt")
```

## Configuration

Edit `configs/default.yaml` to customize:

- Model architecture (SSL type, layer sizes, diffusion steps)
- Training hyperparameters
- Data paths
- Inference settings

## Key Equations

### Forward SDE (Eq. 2)
```
dz_t = -0.5 * β_t * z_t * dt + sqrt(β_t) * dw_t
```

### Reverse SDE (Eq. 3)
```
dz_t = -0.5 * (z_t + ∇log p_t(z_t)) * β_t * dt
```

### Codec Normalization (Eq. 1)
```
ẑ_p → z̃_p = argmin |f_SV(ẑ_p) - f_SV(z̃_p)|
```

## Results

Based on UASpeech corpus evaluation:

| Model                | M12 PER   | F02 PER   | M16 PER   | F04 PER   |
| -------------------- | --------- | --------- | --------- | --------- |
| SV-DSR               | 62.1%     | 49.1%     | 46.5%     | 43.0%     |
| **Diff-DSR (WavLM)** | **61.3%** | **40.3%** | **37.1%** | **33.4%** |

## Citation

```bibtex
@article{chen2025diffdsr,
  title={DiffDSR: Dysarthric Speech Reconstruction Using Latent Diffusion Model},
  author={Chen, Xueyuan and Yang, Dongchao and Wu, Wenxuan and Wu, Minglin and Xu, Jing and Wu, Xixin and Wu, Zhiyong and Meng, Helen},
  journal={arXiv preprint arXiv:2506.00350},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- HuBERT/Wav2Vec2/WavLM: Facebook AI Research
- EnCodec: Meta AI
- NaturalSpeech 2: Microsoft Research
- UASpeech Corpus: University of Illinois
