#Train TanAILite base GPT (causal LM).

import argparse, json, math, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tanailite.model.tanai_gpt import TanAILiteGPT, TanAILiteGPTConfig
from tanailite.tokenizer.tokenizer import TanAILiteTokenizer
from tanailite.train.data_utils import collect_text_lines, list_text_files
from tanailite.utils.checkpoint_io import load_checkpoint, save_checkpoint
from tanailite.utils.runtime import resolve_device, set_seed

def _parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Train TanAILite base GPT")
    p.add_argument("--tokenizer-model", required=True)
    p.add_argument("--train-data",      required=True, help="A text file or a directory.")
    p.add_argument("--data-glob",       default="*.txt")
    p.add_argument("--recursive",       action="store_true")
    p.add_argument("--max-files",       type=int, default=0, help="0 means all files")
    p.add_argument("--max-lines",       type=int, default=0, help="0 means all lines")
    p.add_argument("--min-line-chars",  type=int, default=4)
    p.add_argument("--max-line-chars",  type=int, default=0, help="0 disables char truncation")
    p.add_argument("--max-line-tokens", type=int, default=0, help="0 disables token truncation per line")
    p.add_argument("--stream-add-bos",  action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--stream-add-eos",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-remainder",  action="store_true", help="Keep final short remainder by EOS padding.")

    p.add_argument("--max-seq-len",     type=int, default=1024)
    p.add_argument("--d-model",         type=int, default=512)
    p.add_argument("--n-layers",        type=int, default=8)
    p.add_argument("--n-heads",         type=int, default=8)
    p.add_argument("--mlp-ratio",       type=float, default=4.0)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--rms-eps",         type=float, default=1e-6)
    p.add_argument("--tie-embeddings",  action="store_true")
    p.add_argument("--untie-embeddings", action="store_true")

    p.add_argument("--max-steps",       type=int, default=3000)
    p.add_argument("--batch-size",      type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--eval-ratio",      type=float, default=0.05)
    p.add_argument("--min-eval-blocks", type=int, default=16)
    p.add_argument("--eval-max-batches", type=int, default=50)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.1)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--log-every",       type=int, default=20)
    p.add_argument("--eval-every",      type=int, default=100)
    p.add_argument("--save-every",      type=int, default=200)

    p.add_argument("--device",          default="auto")
    p.add_argument("--seed",            type=int, default=2303)
    p.add_argument("--no-tqdm",         action="store_true")
    p.add_argument("--resume",          default="")
    p.add_argument("--out-ckpt",        required=True)
    p.add_argument("--out-best",        default="")
    p.add_argument("--report-out",      default="")
    return p.parse_args()

def _resolve_files(train_data: Path, args: argparse.Namespace) -> List[Path]:
    if not train_data.exists():
        raise SystemExit(f"train-data not found: {train_data}")
    if train_data.is_file():
        return [train_data]
    return list_text_files(
        train_data,
        pattern   = str(args.data_glob),
        recursive = bool(args.recursive),
        max_files = int(args.max_files),
    )

def _build_token_stream(
    tokenizer       : TanAILiteTokenizer,
    lines           : Sequence[str],
    *,
    max_line_tokens : int,
    add_bos         : bool,
    add_eos         : bool,
    use_tqdm        : bool = False,
) -> List[int]:

    stream: List[int] = []
    line_iter = lines
    if use_tqdm:
        line_iter = tqdm(lines, desc="tokenize-lines", unit="line")

    for line in line_iter:
        ids = tokenizer.encode_ids(
            line,
            add_bos = bool(add_bos),
            add_eos = bool(add_eos),
            max_len = int(max_line_tokens) if int(max_line_tokens) > 0 else None,
        )
        stream.extend(int(i) for i in ids)
    return stream

def _build_lm_blocks(
    stream_ids     : Sequence[int],
    *,
    block_len      : int,
    eos_id         : int,
    keep_remainder : bool,
) -> torch.Tensor:

    if len(stream_ids) < int(block_len):
        raise SystemExit(
            f"token stream too short ({len(stream_ids)}), need at least block_len={block_len}. "
            "Increase corpus size or lower --max-seq-len."
        )

    blocks: List[List[int]] = []
    step = int(block_len)
    for start in range(0, len(stream_ids) - step + 1, step):
        block = list(stream_ids[start : start + step])
        blocks.append(block)

    remainder_start = (len(stream_ids) // step) * step
    remainder       = list(stream_ids[remainder_start:])
    if (keep_remainder and remainder):
        if len(remainder) < step:
            remainder.extend([int(eos_id)] * (step - len(remainder)))
        blocks.append(remainder[:step])

    if not blocks:
        raise SystemExit("no LM blocks produced from token stream")
    return torch.tensor(blocks, dtype=torch.long)

def _split_blocks(
    blocks          : torch.Tensor,
    *,
    eval_ratio      : float,
    min_eval_blocks : int,
    seed            : int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    num  = int(blocks.shape[0])
    idxs = list(range(num))
    rng  = random.Random(int(seed))
    rng.shuffle(idxs)

    eval_count = int(round(num * float(eval_ratio)))
    if (num >= int(min_eval_blocks)):
        eval_count = max(eval_count, int(min_eval_blocks))
    eval_count = min(max(0, eval_count), num - 1) if num > 1 else 0

    eval_ids     = idxs[:eval_count]
    train_ids    = idxs[eval_count:] if eval_count > 0 else idxs
    train_blocks = blocks[train_ids]
    eval_blocks  = blocks[eval_ids] if eval_ids else torch.empty((0, blocks.shape[1]), dtype=torch.long)
    return train_blocks, eval_blocks

def _sample_train_batch(
    blocks     : torch.Tensor,
    *,
    batch_size : int,
    device     : torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    idx       = torch.randint(0, int(blocks.shape[0]), (int(batch_size),), dtype=torch.long)
    batch     = blocks[idx]
    input_ids = batch[:, :-1].to(device)
    labels    = batch[:, 1:].to(device)
    return input_ids, labels

def _compute_lm_loss(model: TanAILiteGPT, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids)
    # next-token CE loss on flattened [B*T, V] vs [B*T].
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

def _evaluate(
    model            : TanAILiteGPT,
    eval_blocks      : torch.Tensor,
    *,
    eval_batch_size  : int,
    eval_max_batches : int,
    device           : torch.device,
) -> Dict[str, float]:

    if int(eval_blocks.shape[0]) == 0:
        return {"eval_loss": 0.0, "eval_ppl": 0.0, "eval_batches": 0}

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i in range(0, int(eval_blocks.shape[0]), int(eval_batch_size)):
            if int(eval_max_batches) > 0 and len(losses) >= int(eval_max_batches):
                break
            batch     = eval_blocks[i : i + int(eval_batch_size)]
            input_ids = batch[:, :-1].to(device)
            labels    = batch[:, 1:].to(device)
            loss      = _compute_lm_loss(model, input_ids, labels)
            losses.append(float(loss.detach().item()))

    if not losses:
        return {"eval_loss": 0.0, "eval_ppl": 0.0, "eval_batches": 0}

    eval_loss = float(sum(losses) / len(losses))
    eval_ppl  = float(math.exp(min(20.0, eval_loss)))
    return {
        "eval_loss"    : eval_loss,
        "eval_ppl"     : eval_ppl,
        "eval_batches" : len(losses),
    }

def _log_line(msg: str, *, use_tqdm: bool) -> None:
    if use_tqdm:
        tqdm.write(msg)
    else:
        print(msg)

def main() -> None:
    args = _parse_args()
    if args.tie_embeddings and args.untie_embeddings:
        raise SystemExit("cannot set both --tie-embeddings and --untie-embeddings")

    use_tqdm = not bool(args.no_tqdm)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    _log_line(f"[TanAILite][base] device={device} seed={int(args.seed)}", use_tqdm=use_tqdm)

    # Get Tokenizer
    tokenizer = TanAILiteTokenizer.from_file(str(args.tokenizer_model))

    # Get Corpus
    files     = _resolve_files(Path(args.train_data), args)
    if (not files):
        raise SystemExit("no train-data files found")
    _log_line(f"[TanAILite][base] files={len(files)}", use_tqdm=use_tqdm)

    lines = collect_text_lines(
        files,
        max_lines      = int(args.max_lines),
        min_chars      = int(args.min_line_chars),
        max_chars      = int(args.max_line_chars),
        show_progress  = use_tqdm,
        progress_desc  = "collect-lines",
    )
    if (not lines):
        raise SystemExit("no valid lines collected from train-data")
    _log_line(f"[TanAILite][base] lines={len(lines)}", use_tqdm=use_tqdm)

    stream_ids = _build_token_stream(
        tokenizer,
        lines,
        max_line_tokens = int(args.max_line_tokens),
        add_bos         = bool(args.stream_add_bos),
        add_eos         = bool(args.stream_add_eos),
        use_tqdm        = use_tqdm,
    )
    if (not stream_ids):
        raise SystemExit("token stream is empty after tokenization")
    _log_line(f"[TanAILite][base] token_stream_len={len(stream_ids)}", use_tqdm=use_tqdm)

    eos_id    = int(tokenizer.eos_id) if int(tokenizer.eos_id) >= 0 else 0
    block_len = int(args.max_seq_len) + 1

    blocks    = _build_lm_blocks(
        stream_ids,
        block_len      = block_len,
        eos_id         = eos_id,
        keep_remainder = bool(args.keep_remainder),
    )

    train_blocks, eval_blocks = _split_blocks(
        blocks,
        eval_ratio      = float(args.eval_ratio),
        min_eval_blocks = int(args.min_eval_blocks),
        seed            = int(args.seed),
    )
    if int(train_blocks.shape[0]) == 0:
        raise SystemExit("train split has zero blocks")
    
    _log_line(f"[TanAILite][base] train_blocks={int(train_blocks.shape[0])} eval_blocks={int(eval_blocks.shape[0])}",use_tqdm=use_tqdm)

    cfg = TanAILiteGPTConfig(
        vocab_size  = int(tokenizer.vocab_size()),
        d_model     = int(args.d_model),
        n_layers    = int(args.n_layers),
        n_heads     = int(args.n_heads),
        max_seq_len = int(args.max_seq_len),
        mlp_ratio   = float(args.mlp_ratio),
        dropout     = float(args.dropout),
        rms_eps     = float(args.rms_eps),
        tie_embeddings=bool(args.tie_embeddings or not args.untie_embeddings),
    )

    model     = TanAILiteGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.95))

    start_step     = 0
    best_eval_loss = float("inf")
    best_step      = 0
    if args.resume:
        payload        = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location="cpu", strict=True,)
        start_step     = int(payload.get("step", 0))
        extra          = payload.get("extra", {}) if isinstance(payload.get("extra"), dict) else {}
        best_eval_loss = float(extra.get("best_eval_loss", best_eval_loss))
        best_step      = int(extra.get("best_step", best_step))

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_best   = Path(args.out_best) if args.out_best else out_ckpt.with_name(out_ckpt.stem + "_best.pt")
    report_out = Path(args.report_out) if args.report_out else out_ckpt.with_name(out_ckpt.stem + "_report.json")

    history: List[Dict[str, object]] = []
    rolling_loss = 0.0
    train_steps = range(start_step + 1, int(args.max_steps) + 1)
    if use_tqdm:
        train_steps = tqdm(
            train_steps,
            total=max(0, int(args.max_steps) - int(start_step)),
            desc="train-base",
            unit="step",
        )

    for step in train_steps:
        model.train()
        input_ids, labels = _sample_train_batch(train_blocks, batch_size=int(args.batch_size), device=device,)

        loss = _compute_lm_loss(model, input_ids, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if (float(args.grad_clip) > 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        rolling_loss += float(loss.detach().item())

        if step % int(args.log_every) == 0 or step == start_step + 1:
            denom = float(args.log_every) if step % int(args.log_every) == 0 else 1.0
            train_loss = rolling_loss / max(1.0, denom)
            train_ppl = float(math.exp(min(20.0, train_loss)))
            _log_line(
                f"[TanAILite][base] step={step} "
                f"train_loss={train_loss:.6f} train_ppl={train_ppl:.4f}",
                use_tqdm=use_tqdm,
            )
            if use_tqdm:
                train_steps.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    train_ppl=f"{train_ppl:.2f}",
                    refresh=False,
                )
            rolling_loss = 0.0

        do_eval = (step % int(args.eval_every) == 0) or (step == int(args.max_steps))
        do_save = (step % int(args.save_every) == 0) or do_eval
        if do_eval:
            eval_res  = _evaluate(model, eval_blocks, eval_batch_size=int(args.eval_batch_size), eval_max_batches=int(args.eval_max_batches), device=device)
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
            _log_line(
                f"[TanAILite][base][eval] step={step} "
                f"eval_loss={eval_loss:.6f} eval_ppl={eval_ppl:.4f}",
                use_tqdm=use_tqdm,
            )
            if use_tqdm:
                train_steps.set_postfix(
                    eval_loss=f"{eval_loss:.4f}",
                    eval_ppl=f"{eval_ppl:.2f}",
                    refresh=False,
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
            "train_data": str(args.train_data),
            "out_ckpt": str(out_ckpt),
            "out_best": str(out_best),
            "report_out": str(report_out),
        },
        "dataset": {
            "files_used": len(files),
            "lines_used": len(lines),
            "token_stream_len": len(stream_ids),
            "total_blocks": int(blocks.shape[0]),
            "train_blocks": int(train_blocks.shape[0]),
            "eval_blocks": int(eval_blocks.shape[0]),
            "block_len_tokens": int(block_len),
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

#PYTHONPATH=/home/sophia/lite python -m tanailite.train.train_base --tokenizer-model "/home/sophia/lite/data/tokenizer/tanai-tokenizer.model" --train-data "/home/sophia/lite/data/corpus" --data-glob "*.txt" --recursive --max-seq-len 1024 --d-model 512 --n-layers 8 --n-heads 8 --mlp-ratio 4.0 --tie-embeddings --max-steps 5000 --batch-size 8 --eval-every 100 --save-every 200 --out-ckpt "/home/sophia/lite/data/model/base_latest.pt" --out-best "/home/sophia/lite/data/model/base_best.pt" --report-out "/home/sophia/lite/data/model/base_report.json"