"""Runtime tokenizer wrapper for TanAILite."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

# Sentencepiece Tokenizer was used for the training.
import sentencepiece as spm

@dataclass(frozen=True)
class TanAILiteTokenizerConfig:
    model_path : str
    add_bos    : bool = True
    add_eos    : bool = True

class TanAILiteTokenizer:
    """Thin SentencePiece wrapper with a stable TanAILite API contract."""

    def __init__(self, cfg: TanAILiteTokenizerConfig):
        self.cfg   = cfg
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"tokenizer model not found: {model_path}")
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))

    @classmethod
    def from_file(
        cls,
        model_path: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> "TanAILiteTokenizer":
        return cls(TanAILiteTokenizerConfig(model_path=model_path, add_bos=add_bos, add_eos=add_eos))

    def encode_ids(
        self,
        text: str,
        *,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
        max_len: int | None = None,
    ) -> List[int]:
        ids = list(
            self._sp.encode(
                text,
                out_type = int,
                add_bos  = self.cfg.add_bos if add_bos is None else bool(add_bos),
                add_eos  = self.cfg.add_eos if add_eos is None else bool(add_eos),
            )
        )
        if max_len is not None and max_len > 0:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._sp.decode(list(ids))

    def vocab_size(self) -> int:
        return int(self._sp.vocab_size())

    @property
    def pad_id(self) -> int:
        return int(self._sp.pad_id())

    @property
    def bos_id(self) -> int:
        return int(self._sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self._sp.eos_id())

    @property
    def unk_id(self) -> int:
        return int(self._sp.unk_id())

    def export_metadata(
        self,
        *,
        path: str | Path,
        tokenizer_version: str = "v1",
        extra: dict | None = None,
    ) -> dict:
        payload = {
            "tokenizer_version": str(tokenizer_version),
            "model_path": str(self.cfg.model_path),
            "vocab_size": int(self.vocab_size()),
            "special_ids": {
                "unk_id": int(self.unk_id),
                "bos_id": int(self.bos_id),
                "eos_id": int(self.eos_id),
                "pad_id": int(self.pad_id),
            },
            "config": {
                "add_bos": bool(self.cfg.add_bos),
                "add_eos": bool(self.cfg.add_eos),
            },
        }
        if extra:
            payload["extra"] = dict(extra)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return payload

    @staticmethod
    def load_metadata(path: str | Path) -> dict:
        meta_path = Path(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"tokenizer metadata not found: {meta_path}")

        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"invalid tokenizer metadata payload: {meta_path}")
        return obj


__all__ = ["TanAILiteTokenizerConfig", "TanAILiteTokenizer"]
