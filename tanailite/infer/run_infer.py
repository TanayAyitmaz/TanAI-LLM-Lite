"""CLI entrypoint for TanAILite inference."""

import argparse, json, re
from pathlib import Path
from typing import Dict, Tuple

import torch

from tanailite.infer.generate      import TanAILiteGenerationConfig, generate_text
from tanailite.model.tanai_gpt     import TanAILiteGPT, TanAILiteGPTConfig
from tanailite.tokenizer.tokenizer import TanAILiteTokenizer
from tanailite.utils.runtime       import resolve_device, set_seed


def _parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Run TanAILite inference")
    
    p.add_argument("--prompt",          required=True, help="Input prompt text.")
    p.add_argument("--tokenizer-model", required=True, help="TanAITokenizer model path.")
    p.add_argument("--model-ckpt",      required=True, help="Model checkpoint path.")

    p.add_argument("--device",          default="auto", help="auto|cpu|cuda[:index]")
    p.add_argument("--seed",            type=int, default=2303)
    p.add_argument("--echo-prompt",     action="store_true")
    p.add_argument("--print-meta",      action="store_true")

    p.add_argument("--prompt-template-name", choices=["plain", "instruct", "chat"], default="plain")
    p.add_argument("--prompt-template",      default="", help="Custom template with '{prompt}' placeholder.")

    p.add_argument("--max-new-tokens",       type=int, default=TanAILiteGenerationConfig.max_new_tokens)
    p.add_argument("--temperature",          type=float, default=TanAILiteGenerationConfig.temperature)
    p.add_argument("--top-k",                type=int, default=TanAILiteGenerationConfig.top_k)
    p.add_argument("--top-p",                type=float, default=TanAILiteGenerationConfig.top_p)
    p.add_argument("--repetition-penalty",   type=float, default=TanAILiteGenerationConfig.repetition_penalty)
    p.add_argument("--eos-id",               type=int, default=-1, help="Set >=0 to override eos token id.")

    # Optional overrides if checkpoint cannot provide all metadata.
    p.add_argument("--vocab-size",           type=int, default=0)
    p.add_argument("--d-model",              type=int, default=0)
    p.add_argument("--n-layers",             type=int, default=0)
    p.add_argument("--n-heads",              type=int, default=0)
    p.add_argument("--max-seq-len",          type=int, default=0)
    p.add_argument("--mlp-ratio",            type=float, default=0.0)
    p.add_argument("--dropout",              type=float, default=0.0)
    p.add_argument("--rms-eps",              type=float, default=0.0)
    p.add_argument("--tie-embeddings",       action="store_true")
    p.add_argument("--untie-embeddings",     action="store_true")
    p.add_argument("--no-strict-load",       action="store_true")
    return p.parse_args()

def _extract_state_dict(payload: object) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    
    if "model" in payload and isinstance(payload["model"], dict):
        model_state = payload["model"]
        meta = payload
    else:
        model_state = payload
        meta = {"model": model_state}
    
    if not any(isinstance(v, torch.Tensor) for v in model_state.values()):
        raise ValueError("checkpoint does not look like a state_dict")
    return model_state, meta

def _infer_cfg_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    inferred: Dict[str, object] = {}
    if "tok_emb.weight" in state:
        inferred["vocab_size"] = int(state["tok_emb.weight"].shape[0])
        inferred["d_model"]    = int(state["tok_emb.weight"].shape[1])
    if "pos_emb.weight" in state:
        inferred["max_seq_len"] = int(state["pos_emb.weight"].shape[0])

    layer_pat = re.compile(r"^blocks\.(\d+)\.")
    layer_ids = []
    for key in state:
        m = layer_pat.match(str(key))
        if m:
            layer_ids.append(int(m.group(1)))

    if layer_ids:
        inferred["n_layers"] = max(layer_ids) + 1

    if "blocks.0.ff_up.weight" in state and "tok_emb.weight" in state:
        d_model = int(state["tok_emb.weight"].shape[1])
        mlp_dim = int(state["blocks.0.ff_up.weight"].shape[0])
        inferred["mlp_ratio"] = float(mlp_dim / max(1, d_model))

    return inferred

def _build_model_cfg(args: argparse.Namespace, state: Dict[str, torch.Tensor], meta: Dict[str, object]) -> TanAILiteGPTConfig:
    cfg       = TanAILiteGPTConfig()
    extra     = meta.get("extra", {}) if isinstance(meta.get("extra"), dict) else {}
    model_cfg = extra.get("model_config", {}) if isinstance(extra.get("model_config"), dict) else {}

    for key in (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "max_seq_len",
        "mlp_ratio",
        "dropout",
        "rms_eps",
        "tie_embeddings",
    ):
        if key in model_cfg:
            setattr(cfg, key, model_cfg[key])

    inferred = _infer_cfg_from_state_dict(state)
    for key, value in inferred.items():
        setattr(cfg, key, value)

    if args.vocab_size > 0:
        cfg.vocab_size = int(args.vocab_size)
    if args.d_model > 0:
        cfg.d_model = int(args.d_model)
    if args.n_layers > 0:
        cfg.n_layers = int(args.n_layers)
    if args.n_heads > 0:
        cfg.n_heads = int(args.n_heads)
    if args.max_seq_len > 0:
        cfg.max_seq_len = int(args.max_seq_len)
    if args.mlp_ratio > 0:
        cfg.mlp_ratio = float(args.mlp_ratio)
    if args.dropout > 0:
        cfg.dropout = float(args.dropout)
    if args.rms_eps > 0:
        cfg.rms_eps = float(args.rms_eps)

    if args.tie_embeddings and args.untie_embeddings:
        raise ValueError("cannot set both --tie-embeddings and --untie-embeddings")
    if args.tie_embeddings:
        cfg.tie_embeddings = True
    if args.untie_embeddings:
        cfg.tie_embeddings = False
    return cfg

# EN: Generally, the syntax <|SYSYEM|><|CONTEXT|><|NOTES|> can be used in SFT (System Function Instruction) training. For the tokenizer to learn this structure, the tokenizer training needs to be customized.
# TR: Genel olarak SFT eğitiminde (Instruction) <|SYSYEM|><|CONTEXT|><|NOTES|> söz dizimi kullanılabilir. Tokenizerın bu yapıyı öğrenmesi için Tokenizer eğitiminin özelleştirilmesi gerekmektedir.
def _resolve_prompt_template(args: argparse.Namespace) -> str | None:
    if args.prompt_template:
        return args.prompt_template
    if args.prompt_template_name == "plain":
        return None
    if args.prompt_template_name == "instruct":
        return "### Instruction:\n{prompt}\n\n### Response:\n"
    return "<|USER|>\n{prompt}\n<|ASSISTANT|>\n"

def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    ckpt_payload     = torch.load(str(Path(args.model_ckpt)), map_location="cpu")
    state_dict, meta = _extract_state_dict(ckpt_payload)
    cfg              = _build_model_cfg(args, state_dict, meta)

    model = TanAILiteGPT(cfg)
    model.load_state_dict(state_dict, strict=not bool(args.no_strict_load))
    model.to(device)
    model.eval()

    tokenizer       = TanAILiteTokenizer.from_file(args.tokenizer_model)
    prompt_template = _resolve_prompt_template(args)
    eos_id          = int(args.eos_id) if int(args.eos_id) >= 0 else None

    gen_cfg = TanAILiteGenerationConfig(
        max_new_tokens     = int(args.max_new_tokens),
        temperature        = float(args.temperature),
        top_k              = int(args.top_k),
        top_p              = float(args.top_p),
        repetition_penalty = float(args.repetition_penalty),
        eos_id             = eos_id, #2
    )

    result = generate_text(
        model           = model,
        tokenizer       = tokenizer,
        prompt          = str(args.prompt),
        generation_cfg  = gen_cfg,
        prompt_template = prompt_template,
        echo_prompt     = bool(args.echo_prompt),
    )
    print(result.output_text)

    if args.print_meta:
        payload = {
            "stop_reason"      : result.stop_reason,
            "prompt_tokens"    : result.prompt_tokens,
            "generated_tokens" : result.generated_tokens,
            "total_tokens"     : result.prompt_tokens + result.generated_tokens,
            "model_config"     : cfg.__dict__,
            "device"           : str(device),
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))

if __name__ == "__main__":
    main()

#python -m tanailite.infer.run_infer --prompt "Merhaba, yolda yürürken" --tokenizer-model "/home/sophia/lite/data/tokenizer/tanai-tokenizer.model" --model-ckpt "/home/sophia/lite/data/model/base_best.pt" --max-new-tokens 128 --temperature 0.8 --top-k 40 --top-p 0.95 --prompt-template-name instruct --print-meta
