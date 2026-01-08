import os


def setup_directories():
    """Create necessary directory structure for DiffDSR."""
    
    directories = [
        'data/librispeech',
        'data/uaspeech',
        'data/vctk',
        'data/libritts',
        'checkpoints',
        'logs',
        'outputs',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\nDirectory structure ready!")
    print("\nNext steps:")
    print("1. Download datasets to data/ directories")
    print("2. Run: python scripts/prepare_normal_codec_set.py")
    print("3. Run: python train/train_content_encoder.py")
    print("4. Run: python train/train_diffusion.py")


if __name__ == "__main__":
    setup_directories()
