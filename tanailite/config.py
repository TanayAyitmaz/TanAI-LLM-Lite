#Shared path-level config for TanAILite training/inference data.

from dataclasses import dataclass

@dataclass
class TanAILitePaths:
    tokenizer_model : str = "./data/tokenizer/tanai-tokenizer.model"
    encoder_ckpt    : str = "./data/encoder/encoder-best.pt"
    base_ckpt       : str = "./data/model/base_best.pt"
    sft_ckpt        : str = "./data/model/sft_best.pt"

__all__ = ["TanAILitePaths"]
