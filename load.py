from pathlib import Path
from cached_path import cached_path
import safetensors.torch

# Local model definition
from model import DualCodecDecoder


def get_model(
    pretrained_path: str = "hf://vanshjjw/autumn-dualcodec-1s-2a-decoder-only-decay-0.9",
    filename: str = "model.safetensors"
):
    """Load a pretrained DualCodec decoder.

    Parameters
    ----------
    pretrained_path : str
        Remote or local path to directory containing the checkpoint.
    filename : str
        Name of the weights file.
    """
    checkpoint_directory = Path(cached_path(pretrained_path))
    checkpoint_file = checkpoint_directory / filename
    
    # enocder parameters only included for backwards compatibility

    model = DualCodecDecoder(
        sample_rate=24000,
        encoder_rates=[4, 5, 6, 8, 2],
        decoder_rates=[2, 8, 6, 5, 4],
        encoder_dim=32,
        decoder_dim=1536,
        n_codebooks=2,
        codebook_size=4096,
        semantic_codebook_size=16384,
        is_causal=True,
        semantic_downsample_factor=4,
    )

    if checkpoint_file.exists():
        safetensors.torch.load_model(model, str(checkpoint_file), strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} not found")

    model.eval()
    return model



if __name__ == "__main__":
    model = get_model()
    
    import torch
    semantic_codes = torch.randint(0, 16384, (1, 1, 1024), dtype=torch.long)
    acoustic_codes = torch.randint(0, 4096, (1, 2, 1024), dtype=torch.long)
    
    with torch.no_grad():
        reconstructed_audio = model.decode_from_codes(semantic_codes, acoustic_codes)
        
    print(reconstructed_audio.shape)
    print(model)
