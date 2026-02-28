"""Shared training data utilities for TanAILite."""

import json, random
from dataclasses import dataclass
from pathlib     import Path
from typing      import List, Sequence, Tuple
from tqdm        import tqdm

@dataclass(frozen=True)
class ContrastiveTriplet:
    case_id : str
    query   : str
    positive: str
    negative: str

def list_text_files(root: Path, pattern: str, recursive: bool, max_files: int) -> List[Path]:
    if recursive:
        files = sorted(p for p in root.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in root.glob(pattern) if p.is_file())
    if max_files > 0:
        files = files[:max_files]
    return files

def collect_text_lines(
    files     : Sequence[Path],
    *,
    max_lines : int,
    min_chars : int,
    max_chars : int,
    show_progress: bool = False,
    progress_desc: str = "collect-lines",
) -> List[str]:

    rows: List[str] = []
    file_iter = files
    if show_progress:
        file_iter = tqdm(files, desc=progress_desc, unit="file")

    for path in file_iter:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = " ".join(raw.strip().split())
                    if len(line) < int(min_chars):
                        continue
                    if int(max_chars) > 0 and len(line) > int(max_chars):
                        line = line[: int(max_chars)]
                    rows.append(line)
                    if int(max_lines) > 0 and len(rows) >= int(max_lines):
                        return rows
        except Exception:
            continue
    return rows

def augment_line(text: str) -> str:
    line = " ".join(text.split())
    line = line.replace(" .", ".").replace(" ,", ",")
    if line and not line.endswith((".", "!", "?")):
        line += "."
    return line

def build_triplets_from_lines(
    lines        : Sequence[str],
    *,
    max_triplets : int,
    seed         : int,
    id_prefix    : str = "corpus",
    show_progress: bool = False,
) -> List[ContrastiveTriplet]:
    if len(lines) < 3:
        return []

    rng = random.Random(int(seed))
    n_lines = len(lines)
    if int(max_triplets) > 0:
        target = min(int(max_triplets), n_lines)
    else:
        target = n_lines

    # Avoid shuffling full corpus index list when only a small subset is needed.
    if target < n_lines:
        idxs = rng.sample(range(n_lines), k=target)
    else:
        idxs = list(range(n_lines))
        rng.shuffle(idxs)

    idx_iter = idxs
    if show_progress:
        idx_iter = tqdm(idxs, desc="build-triplets", unit="triplet")

    out: List[ContrastiveTriplet] = []
    for i, idx in enumerate(idx_iter):
        query = lines[idx]
        positive = augment_line(query)

        neg_idx = idx
        for _ in range(12):
            cand = rng.randint(0, n_lines - 1)
            if cand != idx:
                neg_idx = cand
                break
        if neg_idx == idx:
            continue

        negative = lines[neg_idx]
        out.append(
            ContrastiveTriplet(
                case_id  = f"{id_prefix}_{i}",
                query    = query,
                positive = positive,
                negative = negative,
            )
        )
        if len(out) >= target:
            break
    return out

def load_triplets_jsonl(path: Path, *, max_triplets: int = 0, id_prefix: str = "jsonl") -> List[ContrastiveTriplet]:
    if not path.exists():
        raise FileNotFoundError(f"triplets jsonl not found: {path}")

    out: List[ContrastiveTriplet] = []
    with open(path, "r", encoding="utf-8") as f:

        for i, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue

            row      = json.loads(raw)
            query    = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            negative = str(row.get("negative", "")).strip()
            if not query or not positive or not negative:
                continue
            
            case_id = str(row.get("case_id", f"{id_prefix}_{i}"))
            out.append(
                ContrastiveTriplet(
                    case_id  = case_id,
                    query    = query,
                    positive = positive,
                    negative = negative,
                )
            )
            if int(max_triplets) > 0 and len(out) >= int(max_triplets):
                break
    return out

def split_triplets(
    triplets   : Sequence[ContrastiveTriplet],
    *,
    eval_ratio : float,
    seed       : int,
    min_eval   : int = 16,
) -> Tuple[List[ContrastiveTriplet], List[ContrastiveTriplet]]:

    rows = list(triplets)
    if not rows:
        return [], []

    rng = random.Random(int(seed))
    rng.shuffle(rows)
    n = len(rows)
    eval_count = int(round(n * float(eval_ratio)))

    if n >= int(min_eval):
        eval_count = max(int(min_eval), eval_count)
    eval_count = min(max(0, eval_count), n - 1) if n > 1 else 0

    eval_rows  = rows[:eval_count]
    train_rows = rows[eval_count:] if eval_count > 0 else rows
    return train_rows, eval_rows


__all__ = [
    "ContrastiveTriplet",
    "list_text_files",
    "collect_text_lines",
    "augment_line",
    "build_triplets_from_lines",
    "load_triplets_jsonl",
    "split_triplets",
]
