#Shared checkpoint read/write helpers.

from pathlib import Path
from typing  import Any, Dict
import torch
import torch.nn as nn

# Save CP Helper
def save_checkpoint(
        path      : str | Path,
        *,
        model     : nn.Module,
        optimizer : torch.optim.Optimizer | None = None,
        scheduler : Any | None = None,
        step      : int = 0,
        extra     : Dict[str, Any] | None = None,
    ) -> Path:

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "step"  : int(step),
        "model" : model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload["extra"] = dict(extra)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(out_path)
    return out_path

# Load CP Helper
def load_checkpoint(
        path         : str | Path,
        *,
        model        : nn.Module | None = None,
        optimizer    : torch.optim.Optimizer | None = None,
        scheduler    : Any | None = None,
        map_location : str | torch.device = "cpu",
        strict       : bool = True,
    ) -> Dict[str, Any]:

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload in {ckpt_path}")

    model_state = payload.get("model")
    if model is not None:
        if not isinstance(model_state, dict):
            raise ValueError(f"checkpoint missing model state: {ckpt_path}")
        model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and isinstance(payload.get("optimizer"), dict):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and isinstance(payload.get("scheduler"), dict):
        scheduler.load_state_dict(payload["scheduler"])

    return payload

__all__ = ["save_checkpoint", "load_checkpoint"]
