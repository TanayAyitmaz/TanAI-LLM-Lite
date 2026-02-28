"""Train and evaluate TanAILite (SentencePiece) tokenizer."""

import argparse, json
from datetime import datetime, timezone
from pathlib  import Path
from typing   import List, Sequence

import sentencepiece as spm

from tanailite.tokenizer.tokenizer import TanAILiteTokenizer

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TanAILite tokenizer")

    p.add_argument("--corpus-dir",            required=True, help="Directory with training text files.")
    p.add_argument("--corpus-glob",           default="*.txt", help="Glob pattern used under corpus-dir.")
    p.add_argument("--recursive",             action="store_true", help="Recursively scan sub-folders.")
    p.add_argument("--max-files",             type=int, default=0, help="0 means all files.")
    p.add_argument("--max-lines",             type=int, default=0, help="0 means all lines.")
    p.add_argument("--min-line-chars",        type=int, default=4)
    p.add_argument("--max-line-chars",        type=int, default=0, help="0 disables clipping.")

    p.add_argument("--model-prefix",           required=True, help="Output model prefix (without extension).")
    p.add_argument("--vocab-size",             type=int, default=32000)
    p.add_argument("--model-type",             choices=["unigram", "bpe", "word", "char"], default="unigram")
    p.add_argument("--character-coverage",     type=float, default=0.9995)
    p.add_argument("--input-sentence-size",    type=int, default=5_000_000)
    p.add_argument("--shuffle-input-sentence", action="store_true")
    p.add_argument("--num-threads",            type=int, default=8)

    p.add_argument("--unk-id",                 type=int, default=0)
    p.add_argument("--bos-id",                 type=int, default=1)
    p.add_argument("--eos-id",                 type=int, default=2)
    p.add_argument("--pad-id",                 type=int, default=3)
    p.add_argument("--unk-piece",              default="<unk>")
    p.add_argument("--bos-piece",              default="<s>")
    p.add_argument("--eos-piece",              default="</s>")
    p.add_argument("--pad-piece",              default="<pad>")

    p.add_argument("--user-symbols",           default="", help="Comma-separated list.")
    p.add_argument("--report-out",             default="", help="Tokenizer eval report JSON path.")
    p.add_argument("--metadata-out",           default="", help="Tokenizer metadata JSON path.")
    p.add_argument("--keep-train-text",        action="store_true")
    p.add_argument("--quiet",                  action="store_true")
    return p.parse_args()

def _list_files(root: Path, pattern: str, recursive: bool, max_files: int) -> List[Path]:
    print(f"Reading Files..")
    if recursive:
        files = sorted(p for p in root.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in root.glob(pattern) if p.is_file())
    if max_files > 0:
        files = files[:max_files]
    return files

def _collect_lines(files: Sequence[Path], args: argparse.Namespace) -> List[str]:
    rows: List[str] = []
    max_lines = int(args.max_lines)
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = " ".join(raw.strip().split())
                    if len(line) < int(args.min_line_chars):
                        continue
    
                    if int(args.max_line_chars) > 0 and len(line) > int(args.max_line_chars):
                        line = line[: int(args.max_line_chars)]
                    rows.append(line)
    
                    if (max_lines > 0) and (len(rows) >= max_lines):
                        print(f"Rows {len(rows)}")
                        return rows

        except Exception:
            continue
    return rows

def _train_sentencepiece(train_text_path: Path, args: argparse.Namespace) -> None:
    user_symbols = [s.strip() for s in str(args.user_symbols).split(",") if s.strip()]
    opts = {
        "input": str(train_text_path),
        "model_prefix": str(args.model_prefix),
        "vocab_size": int(args.vocab_size),
        "model_type": str(args.model_type),
        "character_coverage": float(args.character_coverage),
        "input_sentence_size": int(args.input_sentence_size),
        "shuffle_input_sentence": bool(args.shuffle_input_sentence),
        "num_threads": int(args.num_threads),
        "unk_id": int(args.unk_id),
        "bos_id": int(args.bos_id),
        "eos_id": int(args.eos_id),
        "pad_id": int(args.pad_id),
        "unk_piece": str(args.unk_piece),
        "bos_piece": str(args.bos_piece),
        "eos_piece": str(args.eos_piece),
        "pad_piece": str(args.pad_piece),
        "user_defined_symbols": user_symbols,
    }
    spm.SentencePieceTrainer.train(**opts)

def _eval_tokenizer(tokenizer: TanAILiteTokenizer, lines: Sequence[str]) -> dict:
    if not lines:
        return {
            "eval_lines"     : 0,
            "mean_chars"     : 0.0,
            "mean_tokens"    : 0.0,
            "chars_per_token": 0.0,
            "unk_rate"       : 0.0,
        }

    total_chars  = 0
    total_tokens = 0
    unk_tokens   = 0

    for line in lines:
        ids           = tokenizer.encode_ids(line)
        total_chars  += len(line)
        total_tokens += len(ids)
        unk_tokens   += sum(1 for i in ids if int(i) == int(tokenizer.unk_id))

    mean_chars      = float(total_chars) / len(lines)
    mean_tokens     = float(total_tokens) / len(lines)
    chars_per_token = float(total_chars) / max(1, total_tokens)
    unk_rate        = float(unk_tokens) / max(1, total_tokens)

    marker_set      = ["<|USER|>", "<|ASSISTANT|>", "<|SYSTEM|>", "[EMAIL]"]
    marker_checks   = {m: bool(tokenizer.encode_ids(m)) for m in marker_set}

    return {
        "eval_lines"     : len(lines),
        "mean_chars"     : round(mean_chars, 6),
        "mean_tokens"    : round(mean_tokens, 6),
        "chars_per_token": round(chars_per_token, 6),
        "unk_rate"       : round(unk_rate, 8),
        "marker_checks"  : marker_checks,
    }

def main() -> None:
    args       = _parse_args()
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        raise SystemExit(f"corpus-dir not found: {corpus_dir}")

    model_prefix = Path(args.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    files = _list_files(corpus_dir, str(args.corpus_glob), bool(args.recursive), int(args.max_files))
    if (not files):
        raise SystemExit("no corpus files found")

    lines = _collect_lines(files, args)
    if (not lines):
        raise SystemExit("no valid lines collected from corpus")

    train_text_path = model_prefix.parent / f"{model_prefix.name}.train.txt"
    train_text_path.write_text("\n".join(lines), encoding="utf-8")

    if (not args.quiet):
        print(f"[TanAILite] files={len(files)} lines={len(lines)}")
        print(f"[TanAILite] training sentencepiece -> {model_prefix}.model")

    _train_sentencepiece(train_text_path, args)

    model_path = model_prefix.with_suffix(".model")
    if not model_path.exists():
        raise SystemExit(f"tokenizer model not produced: {model_path}")

    tokenizer = TanAILiteTokenizer.from_file(str(model_path))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "vocab_size": int(tokenizer.vocab_size()),
        "corpus": {
            "corpus_dir": str(corpus_dir),
            "corpus_glob": str(args.corpus_glob),
            "recursive": bool(args.recursive),
            "files_used": len(files),
            "lines_used": len(lines),
        },
        "train_args": {
            "vocab_size": int(args.vocab_size),
            "model_type": str(args.model_type),
            "character_coverage": float(args.character_coverage),
            "input_sentence_size": int(args.input_sentence_size),
            "shuffle_input_sentence": bool(args.shuffle_input_sentence),
            "num_threads": int(args.num_threads),
            "user_symbols": [s.strip() for s in str(args.user_symbols).split(",") if s.strip()],
        },
        "metrics": _eval_tokenizer(tokenizer, lines[: min(5000, len(lines))]),
    }

    report_path = Path(args.report_out) if args.report_out else model_prefix.parent / f"{model_prefix.name}.report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    metadata_path = Path(args.metadata_out) if args.metadata_out else model_prefix.parent / f"{model_prefix.name}.metadata.json"
    tokenizer.export_metadata(
        path=metadata_path,
        tokenizer_version="v1",
        extra={
            "report_path"    : str(report_path),
            "trained_at_utc" : report["created_at_utc"],
        },
    )

    if not bool(args.keep_train_text):
        train_text_path.unlink(missing_ok=True)

    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"\nTokenizer model: {model_path}")
    print(f"Report: {report_path}")
    print(f"Metadata: {metadata_path}")

if __name__ == "__main__":
    main()

#python -m lite.tanailite.train.train_tokenizer --corpus-dir "/home/sophia/lite/data/corpus" --corpus-glob "*.txt" --recursive --model-prefix "/home/sophia/lite/data/tokenizer/tanailite" --vocab-size 32000 --model-type unigram --character-coverage 0.9995 --num-threads 16
