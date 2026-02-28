def test_smoke_imports():
    from tanailite.encoder.encoder      import TanAILiteEncoderConfig
    from tanailite.infer.generate       import TanAILiteGenerationConfig
    from tanailite.model.tanai_gpt      import TanAILiteGPTConfig
    from tanailite.tokenizer.tokenizer  import TanAILiteTokenizerConfig
    from tanailite.utils.runtime        import resolve_device
    from tanailite.version              import __version__

    assert __version__
    assert TanAILiteGPTConfig().d_model > 0
    assert TanAILiteTokenizerConfig(model_path="x").add_bos
    assert TanAILiteEncoderConfig().out_dim > 0
    assert TanAILiteGenerationConfig().max_new_tokens > 0
    assert str(resolve_device("cpu")) == "cpu"

def test_checkpoint_roundtrip(tmp_path):
    import torch
    import torch.nn as nn
    from tanailite.utils.checkpoint_io import load_checkpoint, save_checkpoint

    model = nn.Linear(4, 4, bias=False)
    out   = tmp_path / "ckpt.pt"
    save_checkpoint(out, model=model, step=7, extra={"tag": "smoke"})

    restored = nn.Linear(4, 4, bias=False)
    payload  = load_checkpoint(out, model=restored, map_location="cpu")

    assert int(payload["step"]) == 7
    assert payload["extra"]["tag"] == "smoke"
    assert torch.allclose(model.weight, restored.weight)

def test_seed_reproducibility():
    import torch
    from tanailite.utils.runtime import set_seed

    set_seed(2303)
    a = torch.randn(2, 3)
    set_seed(2303)
    b = torch.randn(2, 3)
    assert torch.allclose(a, b)

def test_tanai_gpt_forward_and_cache():
    import torch
    from tanailite.model.tanai_gpt import TanAILiteGPT, TanAILiteGPTConfig

    cfg = TanAILiteGPTConfig(
        vocab_size  = 128,
        d_model     = 64,
        n_layers    = 2,
        n_heads     = 4,
        max_seq_len = 64,
        tie_embeddings=True,
    )
    model     = TanAILiteGPT(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    logits    = model(input_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)

    prefill_logits, kv_cache = model(input_ids[:, :8], use_cache=True)
    next_logits, kv_cache    = model(input_ids[:, 8:16], kv_cache=kv_cache, use_cache=True)
    assert prefill_logits.shape == (2, 8, cfg.vocab_size)
    assert next_logits.shape == (2, 8, cfg.vocab_size)
    assert len(kv_cache) == cfg.n_layers

def test_generate_ids_smoke():
    from tanailite.infer.generate import TanAILiteGenerationConfig, generate_ids
    from tanailite.model.tanai_gpt import TanAILiteGPT, TanAILiteGPTConfig

    cfg = TanAILiteGPTConfig(
        vocab_size  = 96,
        d_model     = 64,
        n_layers    = 2,
        n_heads     = 4,
        max_seq_len = 48,
        tie_embeddings=True,
    )
    model   = TanAILiteGPT(cfg).eval()
    gen_cfg = TanAILiteGenerationConfig(max_new_tokens=8, temperature=0.0, top_k=0, top_p=1.0, eos_id=None)

    output_ids, completion_ids, stop_reason = generate_ids(model, [1, 2, 3, 4], gen_cfg)
    assert len(output_ids) >= 4
    assert len(completion_ids) <= 8
    assert stop_reason in {"max_new_tokens", "max_seq_len", "eos"}

def test_tokenizer_metadata_io(tmp_path):
    import json
    from tanailite.tokenizer.tokenizer import TanAILiteTokenizer

    meta_path = tmp_path / "tanai-tokenizer.meta.json"
    payload   = {"tokenizer_version": "v1", "vocab_size": 32000}
    meta_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = TanAILiteTokenizer.load_metadata(meta_path)
    assert loaded["tokenizer_version"] == "v1"
    assert int(loaded["vocab_size"]) == 32000
