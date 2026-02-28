"""Minimal sentence encoder contract for TanAILite."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TanAILiteEncoderConfig:
    vocab_size  : int = 32000
    d_model     : int = 512
    max_seq_len : int = 1024
    pad_id      : int = 0
    n_layers    : int = 4
    n_heads     : int = 8
    ffn_dim     : int = 2048
    dropout     : float = 0.0
    out_dim     : int = 512

class TanAILiteEncoder(nn.Module):
    #Reference encoder for ECV/RAG style sentence embeddings.

    def __init__(self, cfg: TanAILiteEncoderConfig):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.n_heads,
            dim_feedforward = cfg.ffn_dim,
            dropout         = cfg.dropout,
            activation      = "gelu", #TanAI uses SwiGLU (swish) in the original version.
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder  = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_proj = nn.Linear(cfg.d_model, cfg.out_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        # Attention mask control
        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_id).long()

        pos_ids          = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x                = self.tok_emb(input_ids) + self.pos_emb(pos_ids)
        key_padding_mask = attention_mask == 0
        x                = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Remark: masked mean pooling -> sum(h_i * m_i) / sum(m_i), then L2 normalize.
        mask   = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.out_proj(pooled), dim=-1)

__all__ = ["TanAILiteEncoderConfig", "TanAILiteEncoder"]
