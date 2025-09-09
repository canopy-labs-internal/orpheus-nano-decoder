"""
This is the model architecture for the dualcodec decoder model that is used in orpheus nano.
WARNING: Only the decoder functionality is implemented here.

The model depends on dac.py, quantizer.py, and layers.py.

For loading the weights and using the model, please refer to load.py.
"""

from typing import List, Union

import torch
import torch.nn as nn

from layers import WNConv1d, ConvNeXtBlock
from quantizer import ResidualVectorQuantize
from dac_model import DAC


class DualCodecDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 16384,
        codebook_dim: Union[int, list] = 8,
        semantic_codebook_dim=8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        distill_projection_out_dim=1024,
        convnext_dim=768,
        convnext_layers=4,
        decode_semantic_for_codec=True,
        is_causal=False,
        semantic_downsample_factor=2,
    ):
        self.semantic_downsample_factor = semantic_downsample_factor
        super().__init__()

        self.dac = DAC(
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim,
            decoder_rates,
            n_codebooks,
            codebook_size,
            codebook_dim,
            quantizer_dropout,
            sample_rate,
            distill_projection_out_dim,
            distill=False,
        )
        self.decode_semantic_for_codec = decode_semantic_for_codec
        
        self.semantic_vq = ResidualVectorQuantize(
            convnext_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=semantic_codebook_dim,
        )
        self.convnext_decoder = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=convnext_dim,
                    intermediate_dim=2048,
                    is_causal=is_causal,
                )
                for _ in range(convnext_layers)
            ],
            WNConv1d(
                convnext_dim,
                1024,
                kernel_size=1,
            ),
        )
        if not self.decode_semantic_for_codec:
            assert convnext_dim == 1024

    @torch.no_grad()
    def decode_from_codes(self, semantic_codes, acoustic_codes):
        """both [B, n_q, T]"""
        semantic = self.semantic_vq.from_codes(semantic_codes)[0]
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        audio = self.dac.decode_from_codes(acoustic_codes, semantic)
        return audio

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = 24000,
        n_quantizers: int = None,
        semantic_repr=None,
        bypass_quantize_rate=0.125,
        possibly_no_quantizer=False,
    ):
        pass