#Train TanAILite SFT model with prompt masking.
import argparse, json, math, random, re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from tanailite.model.tanai_gpt     import TanAILiteGPT, TanAILiteGPTConfig
from tanailite.tokenizer.tokenizer import TanAILiteTokenizer
from tanailite.utils.checkpoint_io import load_checkpoint, save_checkpoint
from tanailite.utils.runtime       import resolve_device, set_seed

IGNORE_INDEX = -100

@dataclass(frozen=True)
class SFTSample:
    prompt  : str
    response: str

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TanAILite SFT")

    p.add_argument("--tokenizer-model",     required=True)
    p.add_argument("--base-ckpt",           required=True)
    p.add_argument("--sft-jsonl",           required=True)
    p.add_argument("--out-ckpt",            required=True)
    p.add_argument("--out-best",            default="")
    p.add_argument("--report-out",          default="")
    p.add_argument("--resume",              default="")
    p.add_argument("--no-strict-load",      action="store_true")

    p.add_argument("--max-samples",         type=int, default=0, help="0 means all samples")
    p.add_argument("--max-prompt-tokens",   type=int, default=512)
    p.add_argument("--max-response-tokens", type=int, default=384)
    p.add_argument("--max-seq-len",         type=int, default=0, help="0 -> use model config from checkpoint")
    p.add_argument("--template-name",       choices=["plain", "instruct", "chat"], default="instruct")
    p.add_argument("--template",            default="", help="Custom template with '{prompt}' placeholder.")

    p.add_argument("--max-steps",           type=int, default=1500)
    p.add_argument("--batch-size",          type=int, default=8)
    p.add_argument("--eval-batch-size",     type=int, default=16)
    p.add_argument("--eval-ratio",          type=float, default=0.05)
    p.add_argument("--min-eval-samples",    type=int, default=32)
    p.add_argument("--eval-max-batches",    type=int, default=40)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--weight-decay",        type=float, default=0.05)
    p.add_argument("--grad-clip",           type=float, default=1.0)
    p.add_argument("--log-every",           type=int, default=20)
    p.add_argument("--eval-every",          type=int, default=100)
    p.add_argument("--save-every",          type=int, default=200)

    p.add_argument("--seed",                type=int, default=2303)
    p.add_argument("--device",              default="auto")
    return p.parse_args()

def _extract_state_dict(payload: object) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    if "model" in payload and isinstance(payload["model"], dict):
        return payload["model"], payload
    return payload, {"model": payload}

def _infer_cfg_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    inferred: Dict[str, object] = {}
    if "tok_emb.weight" in state:
        inferred["vocab_size"] = int(state["tok_emb.weight"].shape[0])
        inferred["d_model"] = int(state["tok_emb.weight"].shape[1])
    if "pos_emb.weight" in state:
        inferred["max_seq_len"] = int(state["pos_emb.weight"].shape[0])

    layer_pat = re.compile(r"^blocks\.(\d+)\.")
    ids = []
    for key in state:
        m = layer_pat.match(str(key))
        if m:
            ids.append(int(m.group(1)))
    if ids:
        inferred["n_layers"] = max(ids) + 1

    if "blocks.0.ff_up.weight" in state and "tok_emb.weight" in state:
        d_model = int(state["tok_emb.weight"].shape[1])
        mlp_dim = int(state["blocks.0.ff_up.weight"].shape[0])
        inferred["mlp_ratio"] = float(mlp_dim / max(1, d_model))
    return inferred

def _build_model_cfg(meta: Dict[str, object], state: Dict[str, torch.Tensor], args: argparse.Namespace) -> TanAILiteGPTConfig:
    cfg       = TanAILiteGPTConfig()
    extra     = meta.get("extra", {}) if isinstance(meta.get("extra"), dict) else {}
    from_ckpt = extra.get("model_config", {}) if isinstance(extra.get("model_config"), dict) else {}
 
    for key in cfg.__dict__.keys():
        if key in from_ckpt:
            setattr(cfg, key, from_ckpt[key])

    inferred = _infer_cfg_from_state_dict(state)
    for key, value in inferred.items():
        setattr(cfg, key, value)

    if int(args.max_seq_len) > 0:
        cfg.max_seq_len = int(args.max_seq_len)

    return cfg

def _load_sft_samples(path: Path, *, max_samples: int) -> List[SFTSample]:
    if not path.exists():
        raise FileNotFoundError(f"sft-jsonl not found: {path}")

    out: List[SFTSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue

            prompt   = ""
            response = ""

            if isinstance(row, dict):
                if "prompt" in row and "response" in row:
                    prompt = str(row.get("prompt", "")).strip()
                    response = str(row.get("response", "")).strip()
                elif "instruction" in row and "output" in row:
                    inst = str(row.get("instruction", "")).strip()
                    inp = str(row.get("input", "")).strip()
                    prompt = f"{inst}\n{inp}".strip() if inp else inst
                    response = str(row.get("output", "")).strip()
                elif isinstance(row.get("messages"), list):
                    msgs = row["messages"]
                    context_parts: List[str] = []
                    answer = ""
                    for m in msgs:
                        if (not isinstance(m, dict)):
                            continue

                        role    = str(m.get("role", "")).strip().lower()
                        content = str(m.get("content", "")).strip()

                        if (not content):
                            continue
                        if (role == "assistant"):
                            answer = content
                        else:
                            context_parts.append(content)

                    prompt = "\n".join(context_parts).strip()
                    response = answer.strip()

            if prompt and response:
                out.append(SFTSample(prompt=prompt, response=response))

            if int(max_samples) > 0 and len(out) >= int(max_samples):
                break

    if (not out):
        raise SystemExit("no valid SFT samples parsed from JSONL")
    return out

def _resolve_template(args: argparse.Namespace) -> str | None:
    if args.template:
        if "{prompt}" not in str(args.template):
            raise SystemExit("template must include '{prompt}' placeholder")
        return str(args.template)

    if (args.template_name == "plain"):
        return None

    if (args.template_name == "instruct"):
        return "### Instruction:\n{prompt}\n\n### Response:\n"

    return "<|USER|>\n{prompt}\n<|ASSISTANT|>\n"

def _format_prompt(prompt: str, template: str | None) -> str:
    if template is None:
        return prompt
    return template.format(prompt=prompt)

def _resolve_pad_id(tok: TanAILiteTokenizer) -> int:
    if int(tok.pad_id) >= 0:
        return int(tok.pad_id)
    if int(tok.eos_id) >= 0:
        return int(tok.eos_id)

    return 0

def _encode_sft_sample(
    tok                 : TanAILiteTokenizer,
    sample              : SFTSample,
    *,
    template            : str | None,
    max_prompt_tokens   : int,
    max_response_tokens : int,
    max_seq_len         : int,
    pad_id              : int,
) -> Tuple[List[int], List[int], int, int]:

    prompt_text = _format_prompt(sample.prompt, template)
    prompt_ids  = tok.encode_ids(
        prompt_text,
        add_bos = True,
        add_eos = False,
        max_len = int(max_prompt_tokens) if int(max_prompt_tokens) > 0 else None,
    )
    response_ids = tok.encode_ids(
        sample.response,
        add_bos = False,
        add_eos = True,
        max_len = int(max_response_tokens) if int(max_response_tokens) > 0 else None,
    )
    prompt_ids   = [int(i) for i in prompt_ids]
    response_ids = [int(i) for i in response_ids]

    if not response_ids:
        return [], [], 0, 0

    if len(prompt_ids) + len(response_ids) > int(max_seq_len):
        keep_prompt = max(1, int(max_seq_len) - len(response_ids))
        prompt_ids = prompt_ids[-keep_prompt:]

    if len(prompt_ids) + len(response_ids) > int(max_seq_len):
        keep_response = max(1, int(max_seq_len) - len(prompt_ids))
        response_ids = response_ids[:keep_response]

    ids    = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

    if len(ids) > int(max_seq_len):
        ids    = ids[: int(max_seq_len)]
        labels = labels[: int(max_seq_len)]

    if len(ids) < int(max_seq_len):
        pad_n = int(max_seq_len) - len(ids)
        ids.extend([int(pad_id)] * pad_n)
        labels.extend([IGNORE_INDEX] * pad_n)

    target_tokens = sum(1 for x in labels if int(x) != IGNORE_INDEX)

    return ids, labels, len(prompt_ids), target_tokens

def _build_encoded_dataset(
    tok                 : TanAILiteTokenizer,
    samples             : Sequence[SFTSample],
    *,
    template            : str | None,
    max_prompt_tokens   : int,
    max_response_tokens : int,
    max_seq_len         : int,
    pad_id              : int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:

    rows_ids      : List[List[int]] = []
    rows_labels   : List[List[int]] = []
    prompt_lens   : List[int] = []
    target_counts : List[int] = []

    for s in samples:
        ids, labels, prompt_len, target_tokens = _encode_sft_sample(
            tok,
            s,
            template=template,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
        )
        if (not ids):
            continue

        if (target_tokens <= 0):
            continue

        rows_ids     .append(ids)
        rows_labels  .append(labels)
        prompt_lens  .append(prompt_len)
        target_counts.append(target_tokens)

    if not rows_ids:
        raise SystemExit("no valid encoded SFT samples after filtering")

    ids_t    = torch.tensor(rows_ids, dtype=torch.long)
    labels_t = torch.tensor(rows_labels, dtype=torch.long)
    stats    = {
        "samples"           : len(rows_ids),
        "mean_prompt_tokens": round(float(sum(prompt_lens) / len(prompt_lens)), 6),
        "mean_target_tokens": round(float(sum(target_counts) / len(target_counts)), 6),
        "max_target_tokens" : int(max(target_counts)),
        "min_target_tokens" : int(min(target_counts)),
    }

    return ids_t, labels_t, stats

def _split_indices(n: int, *, eval_ratio: float, min_eval: int, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rng  = random.Random(int(seed))
    rng.shuffle(idxs)
    eval_count = int(round(n * float(eval_ratio)))

    if (n >= int(min_eval)):
        eval_count = max(eval_count, int(min_eval))

    eval_count = min(max(0, eval_count), n - 1) if n > 1 else 0
    eval_ids   = idxs[:eval_count]
    train_ids  = idxs[eval_count:] if eval_count > 0 else idxs

    return train_ids, eval_ids

def _sample_batch(
    ids        : torch.Tensor,
    labels     : torch.Tensor,
    idx_pool   : Sequence[int],
    *,
    batch_size : int,
    device     : torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    pick      = torch.randint(0, len(idx_pool), (int(batch_size),), dtype=torch.long)
    batch_idx = torch.tensor([idx_pool[int(i.item())] for i in pick], dtype=torch.long)
    x         = ids[batch_idx].to(device)
    y         = labels[batch_idx].to(device)

    return x, y

def _sft_loss(model: TanAILiteGPT, input_ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    logits = model(input_ids)
    loss   = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    valid_targets = int((labels[:, 1:] != IGNORE_INDEX).sum().item())
    return loss, valid_targets

def _evaluate(
    model   : TanAILiteGPT,
    ids     : torch.Tensor,
    labels  : torch.Tensor,
    eval_idx: Sequence[int],
    *,
    eval_batch_size : int,
    eval_max_batches: int,
    device          : torch.device,
) -> Dict[str, float]:

    if not eval_idx:
        return {"eval_loss": 0.0, "eval_ppl": 0.0, "eval_batches": 0}

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for start in range(0, len(eval_idx), int(eval_batch_size)):
            if int(eval_max_batches) > 0 and len(losses) >= int(eval_max_batches):
                break

            batch_ids = eval_idx[start : start + int(eval_batch_size)]
            x         = ids[batch_ids].to(device)
            y         = labels[batch_ids].to(device)

            loss, valid_targets = _sft_loss(model, x, y)

            if valid_targets <= 0:
                continue
            losses.append(float(loss.detach().item()))

    if not losses:
        return {"eval_loss": 0.0, "eval_ppl": 0.0, "eval_batches": 0}

    eval_loss = float(sum(losses) / len(losses))
    eval_ppl  = float(math.exp(min(20.0, eval_loss)))
    return {
        "eval_loss"   : eval_loss,
        "eval_ppl"    : eval_ppl,
        "eval_batches": len(losses),
    }

def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    tok      = TanAILiteTokenizer.from_file(str(args.tokenizer_model))
    pad_id   = _resolve_pad_id(tok)
    template = _resolve_template(args)

    base_payload          = torch.load(str(Path(args.base_ckpt)), map_location="cpu")
    base_state, base_meta = _extract_state_dict(base_payload)
    cfg                   = _build_model_cfg(base_meta, base_state, args)
    model                 = TanAILiteGPT(cfg).to(device)
    model.load_state_dict(base_state, strict=not bool(args.no_strict_load))

    samples = _load_sft_samples(Path(args.sft_jsonl), max_samples=int(args.max_samples))
    ids_t, labels_t, ds_stats = _build_encoded_dataset(
        tok,
        samples,
        template=template,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_response_tokens=int(args.max_response_tokens),
        max_seq_len=int(cfg.max_seq_len),
        pad_id=int(pad_id),
    )

    train_idx, eval_idx = _split_indices(
        int(ids_t.shape[0]),
        eval_ratio=float(args.eval_ratio),
        min_eval=int(args.min_eval_samples),
        seed=int(args.seed),
    )
    
    if not train_idx:
        raise SystemExit("Train split is empty")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.95))

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_best   = Path(args.out_best) if args.out_best else out_ckpt.with_name(out_ckpt.stem + "_best.pt")
    report_out = Path(args.report_out) if args.report_out else out_ckpt.with_name(out_ckpt.stem + "_report.json")

    start_step     = 0
    best_eval_loss = float("inf")
    best_step      = 0

    if (args.resume):
        resume_payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location="cpu", strict=not bool(args.no_strict_load),)
        start_step     = int(resume_payload.get("step", 0))
        extra          = resume_payload.get("extra", {}) if isinstance(resume_payload.get("extra"), dict) else {}
        best_eval_loss = float(extra.get("best_eval_loss", best_eval_loss))
        best_step      = int(extra.get("best_step", best_step))

    history: List[Dict[str, object]] = []
    rolling_loss = 0.0

    for step in range(start_step + 1, int(args.max_steps) + 1):
        model.train()
        x, y = _sample_batch(ids_t, labels_t, train_idx, batch_size=int(args.batch_size), device=device)
        loss, valid_targets = _sft_loss(model, x, y)
        if valid_targets <= 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        rolling_loss += float(loss.detach().item())

        if (step % int(args.log_every) == 0) or (step == start_step + 1):
            denom      = float(args.log_every) if step % int(args.log_every) == 0 else 1.0
            train_loss = rolling_loss / max(1.0, denom)
            train_ppl  = float(math.exp(min(20.0, train_loss)))
            print(
                f"[TanAILite][sft] step={step} "
                f"train_loss={train_loss:.6f} train_ppl={train_ppl:.4f} "
                f"valid_targets={valid_targets}"
            )
            rolling_loss = 0.0

        do_eval = (step % int(args.eval_every) == 0) or (step == int(args.max_steps))
        do_save = (step % int(args.save_every) == 0) or do_eval

        if do_eval:
            eval_res  = _evaluate(model, ids_t, labels_t, eval_idx, eval_batch_size=int(args.eval_batch_size), eval_max_batches=int(args.eval_max_batches), device=device)
            eval_loss = float(eval_res["eval_loss"])
            eval_ppl  = float(eval_res["eval_ppl"])
            history.append(
                {
                    "step"        : int(step),
                    "train_loss"  : round(float(loss.detach().item()), 8),
                    "eval_loss"   : round(eval_loss, 8),
                    "eval_ppl"    : round(eval_ppl, 8),
                    "eval_batches": int(eval_res["eval_batches"]),
                }
            )
            print(
                f"[TanAILite][sft][eval] step={step} "
                f"eval_loss={eval_loss:.6f} eval_ppl={eval_ppl:.4f}"
            )

            if (eval_loss > 0) and (eval_loss < best_eval_loss):
                best_eval_loss = eval_loss
                best_step      = int(step)
                save_checkpoint(
                    out_best,
                    model     = model,
                    optimizer = optimizer,
                    step      = step,
                    extra={
                        "best_step"      : int(best_step),
                        "best_eval_loss" : float(best_eval_loss),
                        "model_config"   : cfg.__dict__,
                        "train_args"     : vars(args),
                    },
                )

        if do_save:
            save_checkpoint(
                out_ckpt,
                model     = model,
                optimizer = optimizer,
                step      = step,
                extra={
                    "best_step"      : int(best_step),
                    "best_eval_loss" : float(best_eval_loss),
                    "model_config"   : cfg.__dict__,
                    "train_args"     : vars(args),
                },
            )

    final_eval = history[-1] if history else {
        "step"         : int(start_step),
        "train_loss"   : 0.0,
        "eval_loss"    : 0.0,
        "eval_ppl"     : 0.0,
        "eval_batches" : 0,
    }

    payload = {
        "ok": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "paths": {
            "tokenizer_model": str(args.tokenizer_model),
            "base_ckpt": str(args.base_ckpt),
            "sft_jsonl": str(args.sft_jsonl),
            "out_ckpt": str(out_ckpt),
            "out_best": str(out_best),
            "report_out": str(report_out),
        },
        "dataset": {
            "samples_total": len(samples),
            "samples_encoded": int(ids_t.shape[0]),
            "samples_train": len(train_idx),
            "samples_eval": len(eval_idx),
            "max_seq_len": int(cfg.max_seq_len),
            **ds_stats,
        },
        "model_config": cfg.__dict__,
        "training": {
            "start_step": int(start_step),
            "end_step": int(args.max_steps),
            "best_step": int(best_step),
            "best_eval_loss": round(float(best_eval_loss), 8) if best_eval_loss < float("inf") else None,
            "best_eval_ppl": round(float(math.exp(min(20.0, best_eval_loss))), 8)
            if best_eval_loss < float("inf")
            else None,
        },
        "final_eval": final_eval,
        "history": history,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(payload["final_eval"], ensure_ascii=True, indent=2))
    print(f"\nLatest ckpt: {out_ckpt}")
    print(f"Best ckpt:   {out_best}")
    print(f"Report:      {report_out}")

if __name__ == "__main__":
    main()
