"""Paths and output helpers for the experiment scripts (aggregates only)."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KM = Path(os.environ.get("KM_HOME", str(Path.home() / "km")))
CACHE = KM / "cache"          # per-patient states: never printed, never copied off the cluster
RESULTS = KM / "results"      # aggregates only
for d in (CACHE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
os.umask(0o077)


def out(name, rows):
    """Append aggregate rows (list of dicts) to results/<name>.jsonl and echo them."""
    p = RESULTS / f"{name}.jsonl"
    with open(p, "a") as f:
        for r in rows:
            r = dict(r, written=time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write(json.dumps(r, default=float) + "\n")
            print("RESULT", name, json.dumps(r, default=float), flush=True)


def guarded(main):
    """Run main(); on error print only the exception type and code locations (a message could
    contain a key or a value from the data)."""
    import traceback
    t0 = time.time()
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        where = " <- ".join(f"{fr.filename.split('/')[-1]}:{fr.lineno}" for fr in reversed(tb))
        print(f"FAILED: {type(e).__name__} at {where} (message suppressed)", flush=True)
        raise SystemExit(1)
    print(f"# wall-clock {time.time() - t0:.1f}s", flush=True)
