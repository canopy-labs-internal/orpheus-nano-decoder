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
    
    import dualcodec
    dualcodec_model = dualcodec.get_model("12hz_v1")
    dualcodec_inference = dualcodec.Inference(dualcodec_model)
    
    import torch
    import torchaudio
    from datasets import load_dataset
    dataset = load_dataset("vanshjjw/VA_autumn_processed_clips", split="train", streaming=True)
    
    for example in dataset:
        print(example)
        audio = torch.tensor(example["audio"]["array"], dtype=torch.float32)
        sample_rate = example["audio"]["sampling_rate"]
        audio = torchaudio.functional.resample(audio, sample_rate, 24000)
        audio = audio.reshape(1, 1, -1)
        audio = audio.to("cuda")
        break
    
    model = model.to("cuda")
    
    with torch.no_grad():
        semantic_codes, acoustic_codes = dualcodec_inference.encode(audio, n_quantizers=2)
    
    with torch.no_grad():
        reconstructed_audio = model.decode_from_codes(semantic_codes, acoustic_codes)
    
    
    
    print(model)
    
    torchaudio.save("recond_audio.wav", reconstructed_audio.squeeze(0).cpu(), 24000)