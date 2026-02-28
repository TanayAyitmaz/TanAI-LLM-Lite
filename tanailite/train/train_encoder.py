"""Train TanAILite encoder with contrastive triplets."""

import argparse, json, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tanailite.encoder.encoder import TanAILiteEncoder, TanAILiteEncoderConfig
from tanailite.tokenizer.tokenizer import TanAILiteTokenizer
from tanailite.train.data_utils import (
    ContrastiveTriplet,
    build_triplets_from_lines,
    collect_text_lines,
    list_text_files,
    load_triplets_jsonl,
    split_triplets,
)
from tanailite.utils.checkpoint_io import load_checkpoint, save_checkpoint
from tanailite.utils.runtime import resolve_device, set_seed

CURATED_EVAL_CASES: List[ContrastiveTriplet] = [
    ContrastiveTriplet("weather_istanbul", "Istanbul'da bugun hava nasil?", "Bugun Istanbul hava durumu nasil gorunuyor?", "Python'da dict nasil siralanir?"),
    ContrastiveTriplet("sql_index", "PostgreSQL'de index nasil eklenir?", "Postgres'te bir tabloya index olusturmak icin ne yapmaliyim?", "Bugun deniz suyu sicakligi kac?"),
    ContrastiveTriplet("tokenizer_marker", "<|USER|> kisa ozet uret", "Kullanici mesaji icin kisa bir ozet cikar.", "Linux'ta disk mount etme adimlari nelerdir?"),
    ContrastiveTriplet("pii_mask", "[EMAIL] bilgisini maskele", "E-posta adresini gizleyip [EMAIL] placeholder kullan.", "GPU fan hizini nasil olcerim?"),
    ContrastiveTriplet("error_debug", "Uvicorn 401 hatasini nasil debug ederim?", "401 yetki hatasini log ve route kontrolu ile incele.", "Ispanakli borek tarifini ver."),
    ContrastiveTriplet("rag_route", "RAG route ne zaman acilmali?", "Retrieval yolunu sadece ilgili baglamda aktif etmeliyiz.", "Telefon numarasi formati nasil yazilir?"),
    ContrastiveTriplet("gpu_memory", "CUDA out of memory nasil azaltilir?", "OOM hatasi icin batch-size ve grad-accum ayari dusurulebilir.", "Istanbul'da metro hatti kac km?"),
    ContrastiveTriplet("checkpoint_resume", "Egitimde checkpoint resume ne ise yarar?", "Kesilen egitimi ayni adimdan devam ettirmek icin checkpoint kullanilir.", "JSON dosyasini CSV'ye cevirme komutu nedir?"),
    ContrastiveTriplet("encoder_semantic", "Encoder semantic benzerligi nasil ogrenir?", "Encoder pozitif/negatif ciftlerle anlamsal uzay olusturur.", "BIOS'ta secure boot nasil kapatilir?"),
    ContrastiveTriplet("corpus_clean", "Corpus temizliginde duplicate nasil azaltilir?", "Yinelemeleri hash ve near-duplicate filtresi ile azaltabiliriz.", "Matematikte determinant nasil hesaplanir?"),
    ContrastiveTriplet("chronos_feature", "Chronos projeksiyonu neyi tasiyor?", "Chronos zaman/akislilik sinyallerini modele enjekte eder.", "Docker image nasil push edilir?"),
    ContrastiveTriplet("session_memory", "Session memory neden onemli?", "Modelin onceki turleri hatirlamasi tutarliligi artirir.", "Telefonumun IMEI numarasini nasil ogrenirim?"),
]

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TanAILite contrastive encoder")
    p.add_argument("--tokenizer-model", required=True)
    p.add_argument("--corpus-dir",      default="")
    p.add_argument("--corpus-glob",     default="*.txt")
    p.add_argument("--recursive",       action="store_true")
    p.add_argument("--max-files",       type=int, default=0, help="0 means all files")
    p.add_argument("--max-lines",       type=int, default=0, help="0 means all lines")
    p.add_argument("--min-line-chars",  type=int, default=24)
    p.add_argument("--max-line-chars",  type=int, default=280)
    p.add_argument("--max-triplets",    type=int, default=30000, help="0 means all generated")
    p.add_argument("--triplets-jsonl",  default="")

    p.add_argument("--eval-ratio",      type=float, default=0.1)
    p.add_argument("--min-eval-count",  type=int, default=32)

    p.add_argument("--d-model",         type=int, default=512)
    p.add_argument("--n-layers",        type=int, default=4)
    p.add_argument("--n-heads",         type=int, default=8)
    p.add_argument("--ffn-dim",         type=int, default=2048)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--out-dim",         type=int, default=512)
    p.add_argument("--max-seq-len",     type=int, default=1024)

    p.add_argument("--max-steps",       type=int, default=2000)
    p.add_argument("--batch-size",      type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--temperature",     type=float, default=0.05)
    p.add_argument("--margin",          type=float, default=0.2)
    p.add_argument("--hardneg-lambda",  type=float, default=0.2)
    p.add_argument("--grad-clip",       type=float, default=1.0)

    p.add_argument("--log-every",       type=int, default=20)
    p.add_argument("--eval-every",      type=int, default=100)
    p.add_argument("--save-every",      type=int, default=200)
    p.add_argument("--seed",            type=int, default=2303)
    p.add_argument("--device",          default="auto")
    p.add_argument("--no-tqdm",         action="store_true")

    p.add_argument("--resume",          default="")
    p.add_argument("--out-ckpt",        required=True)
    p.add_argument("--out-best",        default="")
    p.add_argument("--report-out",      default="")

    p.add_argument("--min-pair-acc",    type=float, default=0.60)
    p.add_argument("--min-retrieval-at1", type=float, default=0.25)
    p.add_argument("--min-mrr",         type=float, default=0.40)
    p.add_argument("--no-quality-gate", action="store_true")
    return p.parse_args()

def _sample_batch(cases: Sequence[ContrastiveTriplet], batch_size: int, rng: random.Random) -> List[ContrastiveTriplet]:
    if not cases:
        raise ValueError("cannot sample from empty training set")
    if len(cases) >= batch_size:
        return rng.sample(list(cases), batch_size)
    return [cases[rng.randrange(0, len(cases))] for _ in range(batch_size)]

def _resolve_pad_id(tok: TanAILiteTokenizer) -> int:
    if int(tok.pad_id) >= 0:
        return int(tok.pad_id)
    if int(tok.eos_id) >= 0:
        return int(tok.eos_id)
    return 0

def _tokenize_texts(
    tok         : TanAILiteTokenizer,
    texts       : Sequence[str],
    *,
    max_seq_len : int,
    device      : torch.device,
    pad_id      : int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    ids_rows : List[List[int]] = []
    mask_rows: List[List[int]] = []
    seq_len = int(max_seq_len)

    for text in texts:
        ids  = tok.encode_ids(str(text), add_bos=True, add_eos=True, max_len=seq_len)
        ids  = list(int(i) for i in ids)
        mask = [1] * len(ids)
        if len(ids) < seq_len:
            pad_n = seq_len - len(ids)
            ids.extend([pad_id] * pad_n)
            mask.extend([0] * pad_n)
        ids_rows.append(ids[:seq_len])
        mask_rows.append(mask[:seq_len])

    input_ids      = torch.tensor(ids_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    return input_ids, attention_mask

def _encode_texts(
    model       : TanAILiteEncoder,
    tok         : TanAILiteTokenizer,
    texts       : Sequence[str],
    *,
    max_seq_len : int,
    batch_size  : int,
    device      : torch.device,
    pad_id      : int,
    use_tqdm    : bool = False,
    tqdm_desc   : str = "encode",
) -> torch.Tensor:

    rows: List[torch.Tensor] = []
    model.eval()
    total_chunks = (len(texts) + int(batch_size) - 1) // int(batch_size)
    iterator = range(0, len(texts), int(batch_size))
    if use_tqdm:
        iterator = tqdm(iterator, total=total_chunks, desc=tqdm_desc, unit="batch", leave=False)

    with torch.no_grad():
        for i in iterator:
            chunk = list(texts[i : i + int(batch_size)])
            input_ids, attention_mask = _tokenize_texts(
                tok,
                chunk,
                max_seq_len = max_seq_len,
                device      = device,
                pad_id      = pad_id,
            )
            embs = model(input_ids, attention_mask)
            rows.append(embs.detach().cpu())

    if not rows:
        return torch.empty((0, model.cfg.out_dim), dtype=torch.float32)
    return torch.cat(rows, dim=0)

def _contrastive_loss(
    q: torch.Tensor,
    p: torch.Tensor,
    n: torch.Tensor,
    *,
    temperature: float,
    margin: float,
    hardneg_lambda: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    # in-batch InfoNCE -> CE over similarity matrix q @ p^T / temperature.
    sim_qp  = torch.matmul(q, p.T) / float(temperature)
    targets = torch.arange(sim_qp.shape[0], device=sim_qp.device)
    ce_qp   = F.cross_entropy(sim_qp, targets)
    ce_pq   = F.cross_entropy(sim_qp.T, targets)
    ce_loss = 0.5 * (ce_qp + ce_pq)

    # margin hinge term pulls q-positive above q-negative by at least margin.
    pos_diag    = torch.sum(q * p, dim=-1)
    neg_diag    = torch.sum(q * n, dim=-1)
    margin_loss = F.relu(float(margin) - (pos_diag - neg_diag)).mean()

    total = ce_loss + float(hardneg_lambda) * margin_loss
    details = {
        "ce_loss"     : float(ce_loss.detach().item()),
        "margin_loss" : float(margin_loss.detach().item()),
        "pos_cos"     : float(pos_diag.mean().detach().item()),
        "neg_cos"     : float(neg_diag.mean().detach().item()),
    }
    return total, details

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a * b).item())

def _eval_triplets(cases: Sequence[ContrastiveTriplet], embs: Dict[str, torch.Tensor]) -> Dict[str, object]:
    margins: List[float] = []
    hits = 0
    for c in cases:
        q      = embs[c.query]
        p      = embs[c.positive]
        n      = embs[c.negative]
        margin = _cos(q, p) - _cos(q, n)
        margins.append(margin)
        hits += int(margin > 0.0)

    n = max(1, len(cases))
    return {
        "count"       : len(cases),
        "pair_acc"    : float(hits) / float(n),
        "mean_margin" : float(sum(margins)) / float(n),
    }

def _eval_retrieval(cases: Sequence[ContrastiveTriplet], embs: Dict[str, torch.Tensor]) -> Dict[str, object]:
    positives  = [c.positive for c in cases]
    negatives  = [c.negative for c in cases]
    candidates = positives + negatives
    if not candidates:
        return {"count": 0, "retrieval_at1": 0.0, "mrr": 0.0}

    cand_tensor = torch.stack([embs[t] for t in candidates], dim=0)
    hits   = 0
    rr_sum = 0.0

    for i, c in enumerate(cases):
        q     = embs[c.query].unsqueeze(0)
        sims  = torch.matmul(q, cand_tensor.T).squeeze(0)
        order = torch.argsort(sims, descending=True)
        target_idx = i
        rank  = int((order == target_idx).nonzero(as_tuple=False)[0].item()) + 1
        rr_sum += 1.0 / float(rank)
        hits   += int(int(order[0].item()) == target_idx)

    n = max(1, len(cases))
    return {
        "count"         : len(cases),
        "retrieval_at1" : float(hits) / float(n),
        "mrr"           : float(rr_sum) / float(n),
    }

def _evaluate_suite(
    model         : TanAILiteEncoder,
    tok           : TanAILiteTokenizer,
    holdout_cases : Sequence[ContrastiveTriplet],
    *,
    max_seq_len   : int,
    batch_size    : int,
    device        : torch.device,
    pad_id        : int,
) -> Dict[str, object]:

    eval_cases    = list(CURATED_EVAL_CASES)
    holdout_cases = list(holdout_cases)

    all_cases = list(eval_cases) + list(holdout_cases)
    unique_texts: List[str] = []
    seen = set()
    for c in all_cases:
        for text in (c.query, c.positive, c.negative):
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

    embs_mat = _encode_texts(
        model,
        tok,
        unique_texts,
        max_seq_len = max_seq_len,
        batch_size  = batch_size,
        device      = device,
        pad_id      = pad_id,
        use_tqdm    = True,
        tqdm_desc   = "eval-encode",
    )
    emb_map = {text: embs_mat[i] for i, text in enumerate(unique_texts)}

    curated_triplet   = _eval_triplets(eval_cases, emb_map)
    curated_retrieval = _eval_retrieval(eval_cases, emb_map)

    holdout_triplet   = {"count": 0, "pair_acc": 0.0, "mean_margin": 0.0}
    holdout_retrieval = {"count": 0, "retrieval_at1": 0.0, "mrr": 0.0}
    if holdout_cases:
        holdout_triplet   = _eval_triplets(holdout_cases, emb_map)
        holdout_retrieval = _eval_retrieval(holdout_cases, emb_map)

    return {
        "curated_triplet"   : curated_triplet,
        "curated_retrieval" : curated_retrieval,
        "holdout_triplet"   : holdout_triplet,
        "holdout_retrieval" : holdout_retrieval,
    }

def _quality_pass(metrics: Dict[str, object], args: argparse.Namespace) -> bool:
    source_triplet   = metrics["curated_triplet"]
    source_retrieval = metrics["curated_retrieval"]
    pair_acc         = float(source_triplet["pair_acc"])
    retrieval_at1    = float(source_retrieval["retrieval_at1"])
    mrr              = float(source_retrieval["mrr"])

    return bool(
        pair_acc >= float(args.min_pair_acc)
        and retrieval_at1 >= float(args.min_retrieval_at1)
        and mrr >= float(args.min_mrr)
    )

def _round_metrics(obj: Dict[str, object]) -> Dict[str, object]:
    out = dict(obj)
    for key, value in list(out.items()):
        if isinstance(value, float):
            out[key] = round(float(value), 8)
    return out

def _build_triplet_pool(args: argparse.Namespace, seed: int) -> Tuple[List[ContrastiveTriplet], Dict[str, object]]:
    pool: List[ContrastiveTriplet] = []
    data_notes: Dict[str, object] = {
        "triplets_from_jsonl": 0,
        "triplets_from_corpus": 0,
        "corpus_files_used": 0,
        "corpus_lines_used": 0,
    }

    if args.triplets_jsonl:
        rows = load_triplets_jsonl(Path(args.triplets_jsonl), max_triplets=0)
        pool.extend(rows)
        data_notes["triplets_from_jsonl"] = len(rows)

    if args.corpus_dir:
        root = Path(args.corpus_dir)
        if not root.exists():
            raise SystemExit(f"corpus-dir not found: {root}")
        files = list_text_files(root, str(args.corpus_glob), bool(args.recursive), int(args.max_files))
        lines = collect_text_lines(
            files,
            max_lines = int(args.max_lines),
            min_chars = int(args.min_line_chars),
            max_chars = int(args.max_line_chars),
            show_progress = True,
            progress_desc = "read-corpus",
        )
        cases = build_triplets_from_lines(
            lines,
            max_triplets = int(args.max_triplets),
            seed         = int(seed),
            id_prefix    = "corpus",
            show_progress = True,
        )
        pool.extend(cases)
        data_notes["triplets_from_corpus"] = len(cases)
        data_notes["corpus_files_used"]    = len(files)
        data_notes["corpus_lines_used"]    = len(lines)

    if (not pool):
        raise SystemExit("no triplets collected: provide --triplets-jsonl or --corpus-dir")
    
    return pool, data_notes

def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))
    rng    = random.Random(int(args.seed))
    device = resolve_device(args.device)

    tokenizer = TanAILiteTokenizer.from_file(args.tokenizer_model)
    pad_id    = _resolve_pad_id(tokenizer)

    print("[TanAILite][encoder] preparing triplet pool...")
    pool, data_notes           = _build_triplet_pool(args, int(args.seed))
    train_cases, holdout_cases = split_triplets(
        pool,
        eval_ratio = float(args.eval_ratio),
        seed       = int(args.seed),
        min_eval   = int(args.min_eval_count),
    )
    if not train_cases:
        raise SystemExit("train set is empty after split")
    print(
        f"[TanAILite][encoder] train_cases={len(train_cases)} "
        f"holdout_cases={len(holdout_cases)} curated_cases={len(CURATED_EVAL_CASES)}"
    )

    model_cfg = TanAILiteEncoderConfig(
        vocab_size  = int(tokenizer.vocab_size()),
        d_model     = int(args.d_model),
        max_seq_len = int(args.max_seq_len),
        pad_id      = int(pad_id),
        n_layers    = int(args.n_layers),
        n_heads     = int(args.n_heads),
        ffn_dim     = int(args.ffn_dim),
        dropout     = float(args.dropout),
        out_dim     = int(args.out_dim),
    )
    model     = TanAILiteEncoder(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.95))

    start_step = 0
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model        = model,
            optimizer    = optimizer,
            map_location = "cpu",
            strict       = True,
        )
        start_step = int(payload.get("step", 0))

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_best   = Path(args.out_best) if args.out_best else out_ckpt.with_name(out_ckpt.stem + "_best.pt")
    report_out = Path(args.report_out) if args.report_out else out_ckpt.with_name(out_ckpt.stem + "_report.json")

    history    : List[Dict[str, object]] = []
    best_score = float("-inf")
    best_step  = start_step

    run_loss   = 0.0
    run_ce     = 0.0
    run_margin = 0.0
    run_pos    = 0.0
    run_neg    = 0.0

    step_iter = range(start_step + 1, int(args.max_steps) + 1)
    use_tqdm = not bool(args.no_tqdm)
    if use_tqdm:
        step_iter = tqdm(
            step_iter,
            total=max(0, int(args.max_steps) - int(start_step)),
            desc="train-encoder",
            unit="step",
            dynamic_ncols=True,
        )

    for step in step_iter:
        model.train()
        batch = _sample_batch(train_cases, int(args.batch_size), rng)
        q_texts = [b.query for b in batch]
        p_texts = [b.positive for b in batch]
        n_texts = [b.negative for b in batch]

        q_ids, q_mask = _tokenize_texts(tokenizer, q_texts, max_seq_len=int(args.max_seq_len), device=device, pad_id=pad_id)
        p_ids, p_mask = _tokenize_texts(tokenizer, p_texts, max_seq_len=int(args.max_seq_len), device=device, pad_id=pad_id)
        n_ids, n_mask = _tokenize_texts(tokenizer, n_texts, max_seq_len=int(args.max_seq_len), device=device, pad_id=pad_id)

        q_emb = model(q_ids, q_mask)
        p_emb = model(p_ids, p_mask)
        n_emb = model(n_ids, n_mask)

        loss, details = _contrastive_loss(
            q_emb,
            p_emb,
            n_emb,
            temperature    = float(args.temperature),
            margin         = float(args.margin),
            hardneg_lambda = float(args.hardneg_lambda),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if (float(args.grad_clip) > 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

        optimizer.step()

        run_loss   += float(loss.detach().item())
        run_ce     += float(details["ce_loss"])
        run_margin += float(details["margin_loss"])
        run_pos    += float(details["pos_cos"])
        run_neg    += float(details["neg_cos"])

        if (step % int(args.log_every) == 0) or (step == start_step + 1):
            denom = float(args.log_every) if step % int(args.log_every) == 0 else 1.0
            avg_loss = run_loss / denom
            avg_ce = run_ce / denom
            avg_margin = run_margin / denom
            avg_pos = run_pos / denom
            avg_neg = run_neg / denom
            print(
                f"[TanAILite][encoder] step={step} "
                f"loss={avg_loss:.6f} ce={avg_ce:.6f} "
                f"margin={avg_margin:.6f} pos={avg_pos:.6f} neg={avg_neg:.6f}"
            )
            if use_tqdm:
                step_iter.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ce=f"{avg_ce:.4f}",
                    m=f"{avg_margin:.4f}",
                )
            run_loss   = 0.0
            run_ce     = 0.0
            run_margin = 0.0
            run_pos    = 0.0
            run_neg    = 0.0

        do_eval = (step % int(args.eval_every) == 0) or (step == int(args.max_steps))
        do_save = (step % int(args.save_every) == 0) or do_eval

        metrics: Dict[str, object] | None = None
        if do_eval:
            metrics = _evaluate_suite(
                model,
                tokenizer,
                holdout_cases,
                max_seq_len = int(args.max_seq_len),
                batch_size  = int(args.eval_batch_size),
                device      = device,
                pad_id      = pad_id,
            )
            curated         = metrics["curated_retrieval"]
            curated_triplet = metrics["curated_triplet"]
            score           = float(curated["retrieval_at1"]) + float(curated["mrr"]) + 0.25 * float(curated_triplet["pair_acc"])
            history.append(
                {
                    "step"              : int(step),
                    "curated_triplet"   : _round_metrics(curated_triplet),
                    "curated_retrieval" : _round_metrics(curated),
                    "holdout_triplet"   : _round_metrics(metrics["holdout_triplet"]),
                    "holdout_retrieval" : _round_metrics(metrics["holdout_retrieval"]),
                    "score"             : round(float(score), 8),
                }
            )
            print(
                f"[TanAILite][encoder][eval] step={step} "
                f"pair_acc={float(curated_triplet['pair_acc']):.4f} "
                f"at1={float(curated['retrieval_at1']):.4f} mrr={float(curated['mrr']):.4f} "
                f"score={score:.6f}"
            )
            if use_tqdm:
                step_iter.set_postfix(
                    pair=f"{float(curated_triplet['pair_acc']):.3f}",
                    at1=f"{float(curated['retrieval_at1']):.3f}",
                    mrr=f"{float(curated['mrr']):.3f}",
                )
            if score > best_score:
                best_score = float(score)
                best_step = int(step)
                save_checkpoint(
                    out_best,
                    model     = model,
                    optimizer = optimizer,
                    step      = step,
                    extra={
                        "best_score"   : float(best_score),
                        "model_config" : model_cfg.__dict__,
                        "train_args"   : vars(args),
                    },
                )

        if do_save:
            save_checkpoint(
                out_ckpt,
                model     = model,
                optimizer = optimizer,
                step      = step,
                extra={
                    "best_step"    : int(best_step),
                    "best_score"   : float(best_score),
                    "model_config" : model_cfg.__dict__,
                    "train_args"   : vars(args),
                },
            )

    final_metrics = history[-1] if history else {
        "step": int(start_step),
        "curated_triplet": {"count": 0, "pair_acc": 0.0, "mean_margin": 0.0},
        "curated_retrieval": {"count": 0, "retrieval_at1": 0.0, "mrr": 0.0},
        "holdout_triplet": {"count": 0, "pair_acc": 0.0, "mean_margin": 0.0},
        "holdout_retrieval": {"count": 0, "retrieval_at1": 0.0, "mrr": 0.0},
        "score": 0.0,
    }
    gate_ok = _quality_pass(final_metrics, args) if history else False

    payload = {
        "ok": bool(gate_ok or bool(args.no_quality_gate)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "paths": {
            "tokenizer_model": str(args.tokenizer_model),
            "out_ckpt": str(out_ckpt),
            "out_best": str(out_best),
            "report_out": str(report_out),
        },
        "data": {
            **data_notes,
            "triplets_total": len(pool),
            "triplets_train": len(train_cases),
            "triplets_holdout": len(holdout_cases),
            "curated_eval_cases": len(CURATED_EVAL_CASES),
        },
        "model_config": model_cfg.__dict__,
        "training": {
            "start_step": int(start_step),
            "end_step": int(args.max_steps),
            "best_step": int(best_step),
            "best_score": round(float(best_score), 8) if best_score > float("-inf") else None,
            "quality_gate": {
                "min_pair_acc": float(args.min_pair_acc),
                "min_retrieval_at1": float(args.min_retrieval_at1),
                "min_mrr": float(args.min_mrr),
                "quality_ok": bool(gate_ok),
                "quality_gate_disabled": bool(args.no_quality_gate),
            },
        },
        "final_eval": final_metrics,
        "history": history,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(payload["final_eval"], ensure_ascii=True, indent=2))
    print(f"\nLatest ckpt: {out_ckpt}")
    print(f"Best ckpt:   {out_best}")
    print(f"Report:      {report_out}")

    if not bool(payload["ok"]):
        raise SystemExit(2)

if __name__ == "__main__":
    main()

#PYTHONPATH=/home/sophia/lite python -m tanailite.train.train_encoder --tokenizer-model "/home/sophia/lite/data/tokenizer/tanai-tokenizer.model" --corpus-dir "/home/sophia/lite/data/corpus" --corpus-glob "*.txt" --recursive --max-seq-len 1024 --d-model 512 --n-layers 4 --n-heads 8 --ffn-dim 2048 --out-dim 512 --max-steps 3000 --batch-size 32 --eval-every 100 --save-every 200 --out-ckpt "/home/sophia/lite/data/encoder/encoder_latest.pt" --out-best "/home/sophia/lite/data/encoder/encoder_best.pt" --report-out "/home/sophia/lite/data/encoder/encoder_report.json"
