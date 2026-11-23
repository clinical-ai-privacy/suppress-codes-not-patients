"""Collect the aggregates the figures need from results_cluster/*.jsonl into one small file.

    python3 extract_data.py <results_cluster dir>      ->  km2_figures_data.json (next to this script)

Last written row of every configuration, as code/scripts/report.py does. Aggregates only.
"""
import json
import sys
from pathlib import Path

RC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent / "km2_figures_data.json"


def rows(name):
    return [json.loads(l) for l in (RC / f"{name}.jsonl").read_text().splitlines() if l.strip()]


fl = {(r["db"], r["hierarchy"], r["K"], r["m"], r["L"]): r for r in rows("floor_levels")}
arm = {(r["db"], r["outcome"], r["arm"]): r for r in rows("utility_arms")}
armm = {(r["db"], r["outcome"], r["arm"]): r for r in rows("utility_arms_match")}
ra = {r["key"]: r for r in rows("reaudit")}
D = {"source": "results_cluster", "K": 5}

# Figure 1: refusal floor by top level, official hierarchy, K = 5
D["floor"] = {db: {f"m{m}": [100 * fl[(db, "official", 5, m, L)]["refused_share"] for L in range(4)]
                   for m in (2, 3)} for db in ("mimic", "eicu")}
D["floor"]["levels"] = ["code", "category", "block", "chapter"]

# Figure 2: cost of strict releases, K = 5, m = 3
D["cost"] = {}
for db, oc in (("eicu", "death"), ("mimic", "readmit30"), ("mimic", "death")):
    d = {}
    for name, key in (("floor_block", f"{db}_official2_m3_K5_floor"), ("floor_chapter", f"{db}_official3_m3_K5_floor"),
                      ("sloc_code", f"{db}_official2_m3_K5_sloc0"), ("sloc_block", f"{db}_official2_m3_K5_slocL"),
                      ("b3gs_chapter", f"{db}_official3_m3_K5_b3gs")):
        r = arm[(db, oc, key)]
        d[name] = dict(d=r["d_raw"], lo=r["d_raw_lo"], hi=r["d_raw_hi"], auroc=r["auroc"], key=key)
    D["cost"][f"{db}_{oc}"] = d

# Figure 3: floor against its matched random control, K = 5, m = 3
D["control"] = {}
for db, oc, L in (("eicu", "death", 1), ("eicu", "death", 2), ("eicu", "death", 3),
                  ("mimic", "readmit30", 2), ("mimic", "readmit30", 3)):
    key = f"{db}_official{L}_m3_K5_floor"
    f, c = armm[(db, oc, key)], armm[(db, oc, key + ":randmatch")]
    D["control"][f"{db}_{oc}_L{L}"] = dict(floor=dict(d=f["d_raw"], lo=f["d_raw_lo"], hi=f["d_raw_hi"]),
                                           control=dict(d=c["d_raw"], lo=c["d_raw_lo"], hi=c["d_raw_hi"]))

# Figure 4: whom the release hurts, eICU, block level, K = 5, m = 3
lens = ["1-2", "3-5", "6-10", "11-20", ">=21"]
bands = ["0s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90s"]
f, s = ra["eicu_official2_m3_K5_floor"], ra["eicu_official2_m3_K5_slocL"]
D["whom"] = dict(lens=lens, bands=bands,
                 floor_refused_len=[100 * f[f"refused_len_{x}"] for x in lens],
                 floor_refused_band=[100 * f[f"refused_band_{x}"] for x in bands],
                 sloc_removed_len=[100 * s[f"removed_len_{x}"] for x in lens],
                 sloc_removed_band=[100 * s[f"removed_band_{x}"] for x in bands])

# Figure 5: length-augmented audit, K = 5, m = 3
D["length"] = {}
for db, L in (("eicu", 1), ("eicu", 2), ("mimic", 2), ("mimic", 3)):
    D["length"][f"{db}_L{L}"] = {
        meth: dict(violators=100 * ra[f"{db}_official{L}_m3_K5_{meth}"]["lenaud_violator_share"],
                   unique=100 * ra[f"{db}_official{L}_m3_K5_{meth}"]["lenaud_unique_share"])
        for meth in ("floor", "slocL")}

OUT.write_text(json.dumps(D, indent=1))
print("wrote", OUT.name)
