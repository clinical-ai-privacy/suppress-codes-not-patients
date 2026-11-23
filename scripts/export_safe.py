"""Copy ~/km/results/*.jsonl to ~/km/results_safe/, masking small cells: any integer from 1 to 9
in a field that counts patients or records becomes "<10" (parameters such as L, m, K, rounds,
strata, hospitals and seeds are kept). Only results_safe/ leaves the cluster."""
import json
from pathlib import Path

KEEP = {"L", "m", "K", "rounds", "n_strata", "hospitals", "seed", "frac", "meta_rounds",
        "meta_collapses", "meta_suppressions", "first_size", "strata", "max_codes_ge",
        "meta_cached", "smallest_stratum"}
KEEP_PREFIX = ("hospitals_", "share_", "nodes", "non_nested")
src = Path.home() / "km" / "results"
dst = Path.home() / "km" / "results_safe"
dst.mkdir(exist_ok=True)
masked = 0


def safe(k, v):
    global masked
    if isinstance(v, bool) or not isinstance(v, int):
        return v
    if k in KEEP or k.startswith(KEEP_PREFIX):
        return v
    if 1 <= v <= 9:
        masked += 1
        return "<10"
    return v


for f in sorted(src.glob("*.jsonl")):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rows = [{k: safe(k, v) for k, v in r.items()} for r in rows]
    (dst / f.name).write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"exported {len(list(dst.glob('*.jsonl')))} files, masked {masked} small cells")
