#Slice a large corpus into a distributed plain-text sample.

import argparse, gzip, io, json, random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO

from tqdm import tqdm

def _parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Create a distributed text slice from a large corpus.")
    p.add_argument("--input-dir",           required=True, help="Root corpus folder.")
    p.add_argument("--glob",                default="*.jsonl*", help="File glob under input-dir.")
    p.add_argument("--recursive",           action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-files",           type=int, default=0, help="0 means all files.")
    p.add_argument("--input-format",        choices=["auto", "text", "jsonl"], default="auto")
    p.add_argument("--jsonl-text-key",      default="text", help="Primary text key for JSONL rows.")
    p.add_argument("--jsonl-fallback-keys", default="content,document,body", help="Comma-separated fallback JSONL keys.")
    p.add_argument("--encoding",            default="utf-8")
    p.add_argument("--target-gb",           type=float, default=10.0, help="Target output size in GB.")
    p.add_argument("--allocation",          choices=["weighted", "uniform"], default="weighted")
    p.add_argument("--passes",              type=int, default=1, help="How many passes over files.")
    p.add_argument("--seed",                type=int, default=2303)

    p.add_argument("--out-dir",             required=True, help="Output folder for sliced shards.")
    p.add_argument("--out-prefix",          default="culturax_slice")
    p.add_argument("--shard-max-mb",        type=int, default=512)
    p.add_argument("--report-out",          default="")

    p.add_argument("--min-line-chars",      type=int, default=12)
    p.add_argument("--max-line-chars",      type=int, default=0, help="0 disables truncation.")
    p.add_argument("--strip-empty",         action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run",             action="store_true")
    return p.parse_args()

@dataclass
class SliceStats:
    files_total     : int = 0
    files_processed : int = 0
    lines_seen      : int = 0
    lines_kept      : int = 0
    bytes_written   : int = 0
    shards_written  : int = 0

class ShardWriter:
    def __init__(self, out_dir: Path, out_prefix: str, shard_max_bytes: int, dry_run: bool):
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.shard_max_bytes = int(shard_max_bytes)
        self.dry_run = bool(dry_run)
        self.shard_index = 0
        self.current_bytes = 0
        self.total_bytes = 0
        self.current_fp: TextIO | None = None
        self.shards: List[Dict[str, object]] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _open_next_shard(self) -> None:
        self.shard_index += 1
        path = self.out_dir / f"{self.out_prefix}_{self.shard_index:06d}.txt"
        self.current_bytes = 0
        self.shards.append({"path": str(path), "bytes": 0})
        if self.dry_run:
            self.current_fp = None
        else:
            self.current_fp = open(path, "w", encoding="utf-8")

    def write_line(self, line: str) -> int:
        line_bytes = len(line.encode("utf-8")) + 1
        if self.shard_index == 0 or self.current_bytes + line_bytes > self.shard_max_bytes:
            self.close_current()
            self._open_next_shard()

        if self.current_fp is not None:
            self.current_fp.write(line)
            self.current_fp.write("\n")

        self.current_bytes += line_bytes
        self.total_bytes += line_bytes
        self.shards[-1]["bytes"] = int(self.current_bytes)
        return int(line_bytes)

    def close_current(self) -> None:
        if self.current_fp is not None:
            self.current_fp.close()
            self.current_fp = None

    def close(self) -> None:
        self.close_current()

def _list_files(root: Path, pattern: str, recursive: bool, max_files: int) -> List[Path]:
    if recursive:
        files = sorted(p for p in root.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in root.glob(pattern) if p.is_file())
    if max_files > 0:
        files = files[: int(max_files)]
    return files

def _looks_like_jsonl(path: Path) -> bool:
    name = path.name.lower()
    return (".jsonl" in name) or name.endswith(".json")

def _open_zstd_text(path: Path, encoding: str) -> Iterable[str]:
    try:
        import zstandard as zstd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"zstd file detected but zstandard package is missing: {path}. Install with 'pip install zstandard'.") from exc

    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with io.TextIOWrapper(reader, encoding=encoding, errors="ignore") as text_reader:
                for line in text_reader:
                    yield line

def _iter_raw_lines(path: Path, encoding: str) -> Iterable[str]:
    low = path.name.lower()
    if low.endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding, errors="ignore") as f:
            for line in f:
                yield line
        return
    if low.endswith(".zst") or low.endswith(".zstd"):
        yield from _open_zstd_text(path, encoding)
        return
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            yield line

def _extract_text(
    raw_line      : str,
    *,
    is_jsonl      : bool,
    primary_key   : str,
    fallback_keys : Sequence[str],
) -> str:
    if (not is_jsonl):
        return raw_line
    
    raw_line = raw_line.strip()
    if (not raw_line):
        return ""
    try:
        obj = json.loads(raw_line)
    except Exception:
        return ""
    
    if isinstance(obj, str):
        return obj
    
    if isinstance(obj, dict):
        keys = [primary_key] + [k for k in fallback_keys if k and k != primary_key]
        for key in keys:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return ""

def _normalize_text(text: str, *, min_chars: int, max_chars: int, strip_empty: bool) -> str:
    text = " ".join(text.split())
    if strip_empty and not text:
        return ""
    if len(text) < int(min_chars):
        return ""
    if int(max_chars) > 0 and len(text) > int(max_chars):
        text = text[: int(max_chars)]
    return text

def _compute_quotas(
    files       : Sequence[Path],
    sizes       : Dict[Path, int],
    target_bytes: int,
    allocation  : str,
) -> Dict[Path, int]:
    quotas: Dict[Path, int] = {}
    if not files:
        return quotas

    if allocation == "uniform":
        base = int(target_bytes // len(files))
        rem  = int(target_bytes - (base * len(files)))
        for idx, path in enumerate(files):
            quotas[path] = int(base + (1 if idx < rem else 0))
        return quotas

    total_size = float(sum(max(1, int(sizes[p])) for p in files))
    raw        = [float(target_bytes) * (float(max(1, int(sizes[p]))) / total_size) for p in files]
    floor_vals = [int(x) for x in raw]
    rem        = int(target_bytes - sum(floor_vals))
    frac_idx   = sorted(range(len(files)), key=lambda i: raw[i] - floor_vals[i], reverse=True)

    for i, path in enumerate(files):
        quotas[path] = int(floor_vals[i])
    for j in range(max(0, rem)):
        quotas[files[frac_idx[j % len(frac_idx)]]] += 1
    return quotas

def _slice_corpus(args: argparse.Namespace) -> Dict[str, object]:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"input-dir not found: {input_dir}")

    files = _list_files(
        input_dir,
        pattern   = str(args.glob),
        recursive = bool(args.recursive),
        max_files = int(args.max_files),
    )
    if not files:
        raise SystemExit("no files matched input selection")

    sizes             = {p: int(p.stat().st_size) for p in files}
    total_input_bytes = int(sum(sizes.values()))

    target_bytes    = int(float(args.target_gb) * (1024**3))
    shard_max_bytes = int(args.shard_max_mb) * 1024 * 1024

    quotas    = _compute_quotas(files, sizes, target_bytes, str(args.allocation))
    remaining = dict(quotas)

    fallback_keys = [x.strip() for x in str(args.jsonl_fallback_keys).split(",") if x.strip()]
    stats         = SliceStats(files_total=len(files))
    writer        = ShardWriter(Path(args.out_dir), str(args.out_prefix), shard_max_bytes, bool(args.dry_run))
    global_remaining = int(target_bytes)

    file_order = list(files)
    global_rng = random.Random(int(args.seed))
    global_rng.shuffle(file_order)

    try:
        for pass_idx in range(max(1, int(args.passes))):
            if global_remaining <= 0:
                break

            progress = tqdm(file_order, desc=f"slice pass {pass_idx + 1}/{max(1, int(args.passes))}", unit="file")
            bytes_before_pass = int(writer.total_bytes)

            for file_idx, path in enumerate(progress):
                if global_remaining <= 0:
                    break
                file_need = int(remaining.get(path, 0))
                if file_need <= 0:
                    continue

                stats.files_processed += 1
                file_size  = max(1, int(sizes[path]))
                pass_boost = 1.0 + (0.5 * float(pass_idx))
                keep_prob  = min(1.0, pass_boost * (float(file_need) / float(file_size)))

                if str(args.input_format) == "jsonl":
                    is_jsonl = True
                elif str(args.input_format) == "text":
                    is_jsonl = False
                else:
                    is_jsonl = _looks_like_jsonl(path)

                file_rng = random.Random(f"{args.seed}:{pass_idx}:{file_idx}:{path}")
                file_written = 0

                for raw in _iter_raw_lines(path, str(args.encoding)):
                    if global_remaining <= 0 or file_written >= file_need:
                        break

                    stats.lines_seen += 1
                    txt = _extract_text(raw, is_jsonl=is_jsonl, primary_key=str(args.jsonl_text_key), fallback_keys=fallback_keys)
                    txt = _normalize_text(txt, min_chars=int(args.min_line_chars), max_chars=int(args.max_line_chars), strip_empty=bool(args.strip_empty))
                    if not txt:
                        continue

                    if file_rng.random() > keep_prob:
                        continue

                    line_bytes = len(txt.encode("utf-8")) + 1
                    if line_bytes > global_remaining:
                        continue
                    if line_bytes > (file_need - file_written):
                        continue

                    writer.write_line(txt)
                    stats.lines_kept    += 1
                    stats.bytes_written += int(line_bytes)
                    global_remaining    -= int(line_bytes)
                    file_written        += int(line_bytes)

                    if global_remaining <= 0:
                        break

                remaining[path] = max(0, int(file_need - file_written))

                progress.set_postfix(written_gb=round(float(writer.total_bytes) / float(1024**3), 3), target_gb=round(float(target_bytes) / float(1024**3), 3))

            progress.close()

            if writer.total_bytes == bytes_before_pass:
                break
    finally:
        writer.close()

    stats.shards_written = int(writer.shard_index)
    achieved_ratio       = float(stats.bytes_written) / float(target_bytes) if target_bytes > 0 else 0.0

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(stats.bytes_written > 0),
        "target_bytes": int(target_bytes),
        "written_bytes": int(stats.bytes_written),
        "achieved_ratio": round(float(achieved_ratio), 8),
        "input_bytes_total": int(total_input_bytes),
        "input_selection": {
            "input_dir": str(input_dir),
            "glob": str(args.glob),
            "recursive": bool(args.recursive),
            "files_total": int(stats.files_total),
            "files_processed": int(stats.files_processed),
            "allocation": str(args.allocation),
            "passes": int(args.passes),
            "seed": int(args.seed),
            "input_format": str(args.input_format),
        },
        "output": {
            "out_dir": str(args.out_dir),
            "out_prefix": str(args.out_prefix),
            "shard_max_mb": int(args.shard_max_mb),
            "shards_written": int(stats.shards_written),
            "shards": writer.shards,
            "dry_run": bool(args.dry_run),
        },
        "line_stats": {
            "lines_seen": int(stats.lines_seen),
            "lines_kept": int(stats.lines_kept),
            "min_line_chars": int(args.min_line_chars),
            "max_line_chars": int(args.max_line_chars),
        },
    }
    return report

def main() -> None:
    args        = _parse_args()
    report      = _slice_corpus(args)
    report_path = Path(args.report_out) if args.report_out else Path(args.out_dir) / f"{args.out_prefix}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"\nReport: {report_path}")

if __name__ == "__main__":
    main()

# python lite/tanailite/utils/corpus_slicer.py --input-dir /home/tanai/ai/data/corpus/culturax --glob "*.txt" --recursive --input-format text --target-gb 10 --allocation weighted --passes 1 --seed 2303 --out-dir /home/tanai/ai/data/corpus/culturax --out-prefix culturax_10gb --shard-max-mb 10240 --min-line-chars 16 --max-line-chars 0
