from pathlib import Path
from cached_path import cached_path
import safetensors.torch
import hydra
from hydra import initialize_config_dir, compose

from model import DualCodec, DAC 

_MODEL2WEIGHTS = {"12hz_simple": "model.safetensors"}
_MODEL2CFG = {"12hz_simple": "decoder_12hz_3vq.yaml"}


def get_model(model_id: str = "12hz_simple", pretrained_path: str = "hf://vanshjjw/autumn-dualcodec-1s-2a-decoder-only-decay-0.9", fname: str | None = None):
    pretrained_path = Path(cached_path(pretrained_path))
    if fname is None:
        fname = _MODEL2WEIGHTS[model_id]

    conf_dir = Path(__file__).parent / "conf/model"
    with initialize_config_dir(version_base="1.3", config_dir=str(conf_dir)):
        cfg = compose(config_name=_MODEL2CFG[model_id])
        model = hydra.utils.instantiate(cfg.model)

    ckpt = pretrained_path / fname
    if ckpt.exists():
        safetensors.torch.load_model(model, str(ckpt), strict=False)
        
    model.eval()
    return model


if __name__ == "__main__":
    model = get_model()
    
    # Load original DualCodec for encoding
    import dualcodec
    from dualcodec import get_model as get_original_model
    original_model = get_original_model("12hz_v1")
    dualcodec_inference = dualcodec.Inference(original_model)
    
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