#Run lightweight smoke checks without external test dependencies.

import tempfile
from pathlib import Path

def main() -> int:
    import torch
    import torch.nn as nn

    from tanailite.encoder.encoder import TanAILiteEncoderConfig
    from tanailite.infer.generate  import TanAILiteGenerationConfig, generate_ids
    from tanailite.model.tanai_gpt import TanAILiteGPT, TanAILiteGPTConfig
    from tanailite.tokenizer.tokenizer import TanAILiteTokenizerConfig
    from tanailite.utils.checkpoint_io import load_checkpoint, save_checkpoint
    from tanailite.utils.runtime import resolve_device, set_seed
    from tanailite.version import __version__

    if not __version__:
        raise RuntimeError("empty version")
    if TanAILiteGPTConfig().d_model <= 0:
        raise RuntimeError("invalid model config")
    if not TanAILiteTokenizerConfig(model_path="dummy").add_bos:
        raise RuntimeError("invalid tokenizer config")
    if TanAILiteEncoderConfig().out_dim <= 0:
        raise RuntimeError("invalid encoder config")
    if TanAILiteGenerationConfig().max_new_tokens <= 0:
        raise RuntimeError("invalid generation config")
    if str(resolve_device("cpu")) != "cpu":
        raise RuntimeError("device resolver mismatch")

    model_cfg = TanAILiteGPTConfig(
        vocab_size     = 128,
        d_model        = 64,
        n_layers       = 2,
        n_heads        = 4,
        max_seq_len    = 64,
        tie_embeddings = True,
    )
    model_core  = TanAILiteGPT(model_cfg).eval()
    demo_ids    = torch.randint(0, model_cfg.vocab_size, (2, 16), dtype=torch.long)
    demo_logits = model_core(demo_ids)

    if demo_logits.shape != (2, 16, model_cfg.vocab_size):
        raise RuntimeError("model forward shape mismatch")

    _, cache       = model_core(demo_ids[:, :8], use_cache=True)
    step_logits, _ = model_core(demo_ids[:, 8:16], kv_cache=cache, use_cache=True)
    if step_logits.shape != (2, 8, model_cfg.vocab_size):
        raise RuntimeError("kv-cache forward shape mismatch")

    out_ids, comp_ids, stop_reason = generate_ids(model_core, [1, 2, 3, 4], TanAILiteGenerationConfig(max_new_tokens=8, temperature=0.0))
    if len(out_ids) < 4 or len(comp_ids) > 8:
        raise RuntimeError("generate_ids shape mismatch")
    
    if stop_reason not in {"max_new_tokens", "max_seq_len", "eos"}:
        raise RuntimeError("invalid stop_reason")

    set_seed(2303)
    a = torch.randn(2, 3)
    set_seed(2303)
    b = torch.randn(2, 3)
    if not torch.allclose(a, b):
        raise RuntimeError("seed reproducibility failed")

    model    = nn.Linear(4, 4, bias=False)
    restored = nn.Linear(4, 4, bias=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke_ckpt.pt"
        save_checkpoint(ckpt_path, model=model, step=7, extra={"tag": "smoke"})
        payload   = load_checkpoint(ckpt_path, model=restored, map_location="cpu")

        if int(payload["step"]) != 7:
            raise RuntimeError("checkpoint step mismatch")

        if payload.get("extra", {}).get("tag") != "smoke":
            raise RuntimeError("checkpoint extra mismatch")

        if not torch.allclose(model.weight, restored.weight):
            raise RuntimeError("checkpoint model mismatch")

    print("Smoke checks passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#Test için sadece Run edin. python -m tanailite.tools.run_smoke
