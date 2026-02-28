"""TanAILite GPT core.
This module intentionally keeps the whole decoder core in one script:
- TanAIRMSNorm
- TanAILiteBlock
- TanAILiteGPTConfig
- TanAILiteGPT
"""

from dataclasses import dataclass
from math import sqrt
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KVPair = Tuple[torch.Tensor, torch.Tensor]

# RMSNorm - TanAI uses AdaRMSNorm and ada_proj. This version is a simplified version.
class TanAIRMSNorm(nn.Module):
    #Root Mean Square LayerNorm without mean subtraction.

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remark: y = x / sqrt(mean(x^2) + eps) * gamma
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class TanAILiteBlock(nn.Module):
    #Pre-norm decoder block with causal self-attention + MLP.

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rms_eps: float = 1e-6,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model  = int(d_model)
        self.n_heads  = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.mlp_dim  = int(round(self.d_model * float(mlp_ratio)))

        self.norm_attn = TanAIRMSNorm(self.d_model, eps=rms_eps)
        self.q_proj    = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj    = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj    = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj    = nn.Linear(self.d_model, self.d_model, bias=False)

        self.norm_mlp  = TanAIRMSNorm(self.d_model, eps=rms_eps)
        self.ff_up     = nn.Linear(self.d_model, self.mlp_dim, bias=False)
        self.ff_down   = nn.Linear(self.mlp_dim, self.d_model, bias=False)
        self.dropout   = nn.Dropout(float(dropout))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, head_dim = x.shape
        out = x.transpose(1, 2).contiguous()
        return out.view(bsz, seq_len, n_heads * head_dim)

    def _build_causal_mask(self, tgt_len: int, total_len: int, past_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(past_len, past_len + tgt_len, device=device).unsqueeze(1)
        k_pos = torch.arange(total_len, device=device).unsqueeze(0)
        # Remark: Query at absolute position i can only attend keys <= i.
        return k_pos <= q_pos

    def _expand_key_mask(self, attention_mask: torch.Tensor, total_len: int) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be 2D [batch, seq]")
        
        key_mask = attention_mask.to(torch.bool)
        if key_mask.shape[1] == total_len:
            return key_mask
        if key_mask.shape[1] > total_len:
            raise ValueError(f"attention_mask seq={key_mask.shape[1]} cannot exceed key length={total_len}")
        
        pad = torch.ones((key_mask.shape[0], total_len - key_mask.shape[1]), device=key_mask.device, dtype=torch.bool)
        return torch.cat([pad, key_mask], dim=1)

    def forward(
        self,
        x              : torch.Tensor,
        *,
        attention_mask : torch.Tensor | None = None,
        past_kv        : KVPair | None = None,
        use_cache      : bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, KVPair]:

        residual = x
        h = self.norm_attn(x)
        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))

        past_len = 0
        if (past_kv is not None):
            past_k, past_v = past_kv
            past_len       = int(past_k.shape[2])
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scale      = 1.0 / sqrt(self.head_dim)
        att_scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        tgt_len   = q.shape[2]
        total_len = k.shape[2]
        causal    = self._build_causal_mask(tgt_len, total_len, past_len, x.device)

        mask_value = -1e4 if att_scores.dtype in (torch.float16, torch.bfloat16) else -1e9
        att_scores = att_scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), mask_value)

        if (attention_mask is not None):
            key_mask = self._expand_key_mask(attention_mask, total_len)
            att_scores = att_scores.masked_fill(~key_mask.unsqueeze(1).unsqueeze(1), mask_value)

        # softmax(QK^T / sqrt(d_k))V is computed in fp32 then cast back for stability.
        att_probs = torch.softmax(att_scores.float(), dim=-1).to(q.dtype)
        att_probs = torch.nan_to_num(att_probs, nan=0.0, posinf=0.0, neginf=0.0)
        att_out   = torch.matmul(att_probs, v)
        att_out   = self._merge_heads(att_out)
        x         = residual + self.dropout(self.o_proj(att_out))

        residual = x
        h = self.norm_mlp(x)
        h = self.ff_up(h)
        h = F.gelu(h)
        h = self.ff_down(h)
        x = residual + self.dropout(h)

        if use_cache:
            return x, (k, v)
        return x

@dataclass
class TanAILiteGPTConfig:
    vocab_size  : int = 32000
    d_model     : int = 512
    n_layers    : int = 8
    n_heads     : int = 8
    max_seq_len : int = 1024
    mlp_ratio   : float = 4.0
    dropout     : float = 0.0
    rms_eps     : float = 1e-6
    tie_embeddings: bool = True
# EN: You can increase the number of model parameters by increasing the configuration values. However, this means higher GPU VRAM usage. After increasing the number of parameters, the Encoder, Base Model, and SFT must be retrained.
# TR: Config değerlerini yükselterek model parametre sayısını arttırabilirsiniz. Fakat bu durum yüksek GPU VRAM kullanımı anlamına gelmektedir. Parametre sayısı arttırıldıktan sonra Encoder, Base Model ve SFT yeniden eğitilmelidir.

class TanAILiteGPT(nn.Module):
    #Causal decoder-only GPT model used by TanAILite.

    def __init__(self, cfg: TanAILiteGPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks  = nn.ModuleList(
            [
                TanAILiteBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    rms_eps=cfg.rms_eps,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm    = TanAIRMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def _resolve_past_len(self, kv_cache: List[KVPair | None] | None) -> int:
        if not kv_cache:
            return 0
        for item in kv_cache:
            if item is not None:
                return int(item[0].shape[2])
        return 0

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor | None = None,
        kv_cache       : List[KVPair | None] | None = None,
        use_cache      : bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[KVPair | None]]:

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D [batch, seq]")

        bsz, seq_len = input_ids.shape
        if (kv_cache is not None) and (len(kv_cache) != self.cfg.n_layers):
            raise ValueError(f"kv_cache must have {self.cfg.n_layers} entries")

        past_len  = self._resolve_past_len(kv_cache)
        total_len = past_len + seq_len
        if (total_len > self.cfg.max_seq_len):
            raise ValueError(f"total sequence length ({total_len}) exceeds max_seq_len ({self.cfg.max_seq_len})")

        pos_ids = torch.arange(past_len, total_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x       = self.tok_emb(input_ids) + self.pos_emb(pos_ids)

        if (kv_cache is None):
            kv_cache = [None] * self.cfg.n_layers

        new_cache: List[KVPair | None] = []

        for i, block in enumerate(self.blocks):
            out = block(
                x,
                attention_mask = attention_mask,
                past_kv        = kv_cache[i],
                use_cache      = use_cache,
            )
            if use_cache:
                x, layer_kv = out
                new_cache.append(layer_kv)
            else:
                x = out

        x      = self.norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_cache
        return logits

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        count  = sum(p.numel() for p in params)
        # Tied embeddings share one tensor, so they are naturally counted once.
        return int(count)


__all__ = [
    "KVPair",
    "TanAIRMSNorm",
    "TanAILiteBlock",
    "TanAILiteGPTConfig",
    "TanAILiteGPT",
]
