# Suppress Codes, Not Patients

Code and aggregate results for the paper *"Suppress Codes, Not Patients: What k^m-Anonymity Costs on
Complete Diagnosis Histories"* by Pavlo Zamroz, Sandrine Vaton and Kassem Kallas.

The paper asks what k^m-anonymity costs when the released record is the complete diagnosis history
of a patient, on MIMIC-IV and eICU-CRD. This repository holds the `kmrel` package, the scripts that
produced every number, the aggregate results, the figure scripts and a checker that compares the
tables of the manuscript with the results. It holds no patient-level data.

## Requirements

Python 3.10 or later with numpy, scipy, pandas, scikit-learn, pytest and tabulate. The results were
produced with numpy 1.26, scipy 1.13, pandas 2.2 and scikit-learn 1.5. The figure scripts also need
matplotlib and Pillow.

```
pip install numpy scipy pandas scikit-learn pytest tabulate matplotlib pillow
```

## Layout

| path | what |
|---|---|
| `kmrel/hierarchy.py` | the truncation tree and the official ICD-9-CM (2011) / ICD-10-CM (FY2026) tree: code, category, block, chapter, root; built from the public CDC files in `hierarchy_data/` |
| `kmrel/data.py` | cohorts (MIMIC-IV, eICU-CRD), strata (sex x 10-year band), outcome tables |
| `kmrel/core.py` | the Release object, the strict (exact string) and loose (ancestor) audits, the exact minimum-support kernel for m <= 4, the audit summary |
| `kmrel/algos.py` | floor, Apriori (Terrovitis et al. 2008) and Apriori on the floor, B3 / B3g, S-loc, the hybrids B3s / B3gs (with the suppression price `supp_w`) |
| `kmrel/attack.py` | the most-specific-match attack |
| `kmrel/utility.py` | TF-IDF with logistic regression or a random forest, 25 patient-grouped splits, Nadeau-Bengio intervals, the `:nosupp`, `:held` and `:randmatch` control arms |
| `kmrel/extra.py` | length-augmented audit, visible-removal share, removal profile by subgroup, count-query error |
| `scripts/check_cohorts.py` | cohort and hierarchy checks |
| `scripts/floor_grid.py` | the refusal floor by level, K, m, population fraction, strata width and per hospital |
| `scripts/releases.py` | build, audit (both counts), attack and extra measurements for a list of releases; `--strata band|hospital|length`, `--supp-w` |
| `scripts/reaudit.py` | the extra measurements on cached releases |
| `scripts/utility_run.py` | the utility protocol for a list of release keys; `--model lr|rf` |
| `scripts/export_safe.py` | copies `results/*.jsonl` to `results_safe/` with counts 1-9 masked (the only thing that leaves the machine with the data) |
| `scripts/report.py` | builds `RESULTS.md` from `results_cluster/` (runs anywhere) |
| `cluster/run.slurm` | generic SLURM runner |
| `hierarchy_data/` | the download script and the checksums of the two public ICD files; see `DATA.md` |
| `tests/` | synthetic-data tests: kernel against brute force, audits against the definitions, floor = greatest fixed point, every algorithm valid under both counts, the extra measurements |
| `results_cluster/` | the aggregate results, one `.jsonl` per result type, counts 1-9 masked |
| `RESULTS.md` | every table, built from `results_cluster/` |
| `figures/` | the scripts of the five figures of the paper and the small data file they read |
| `verification/check_tables.py` | compares every number in the tables of the manuscript with `results_cluster/` |

## Data

Nothing patient-level is here. `DATA.md` says how to obtain the three inputs: the two public ICD
hierarchy files (CDC links, checksums, a download script) and the two PhysioNet databases, which need
your own credential and a signed data use agreement.

The code reads the PhysioNet tables under `$KM_DATA` (default `~/km_data`). Caches and results are
written under `$KM_HOME` (default `~/km`): `cache/` holds one `.npz` per release state, which is
patient-level and must stay on the machine with the data, and `results/` holds one `.jsonl` per result
type.

## Tests

```
python3 -m pytest -q tests/
```

The tests use synthetic cohorts and need no data. One test is skipped until the ICD hierarchy files
are downloaded.

## Reproducing the results

```
# the refusal floor
sbatch -J floor_mimic cluster/run.slurm scripts/floor_grid.py --db mimic
sbatch -J floor_eicu  cluster/run.slurm scripts/floor_grid.py --db eicu

# releases at K = 5, m = 3, official tree, block level
sbatch -J rel cluster/run.slurm scripts/releases.py --db eicu --hier official --L 2 --m 3 --K 5 \
    --methods floor,apriori,b3,b3g,slocL,sloc0,b3s,b3gs

# utility for those releases (keys = cache names printed by releases.py)
sbatch -J util cluster/run.slurm scripts/utility_run.py --db eicu --outcome death --levels official \
    --keys eicu_official2_m3_K5_floor,eicu_official2_m3_K5_slocL,eicu_official2_m3_K5_b3gs

# export the aggregates, then build the summary
python3 scripts/export_safe.py
python3 scripts/report.py
```

Without SLURM, run the same scripts with `python3` from this directory. Cache keys follow
`(mimic|eicu)_(official|trunc)L_m{m}_K{K}[_hosp|_len]_{method}[_w{w}]`.

## The two earlier local methods

The releases `A_import` and `b2_import` are the states produced by two earlier local methods, which
the paper audits and does not propose. `scripts/releases.py --import-earlier` reads them from state
files under `~/km/earlier_states/` and audits them unchanged. Those files are patient-level and are
not distributed, and the code that built them is not part of this repository. The rows of these two
releases in `results_cluster/` therefore cannot be regenerated from this repository alone.

## Figures

```
python3 figures/data/extract_data.py results_cluster     # writes figures/data/km2_figures_data.json
python3 figures/gen_fig1_floor.py                         # and gen_fig2 ... gen_fig5
```

Each script writes a vector PDF, a 300 dpi PNG and a 600 dpi TIFF next to itself. The font is Roboto,
bundled in `figures/fonts/` under the Apache License 2.0.

## Checking the tables of the manuscript

```
python3 -I verification/check_tables.py <manuscript.tex> results_cluster [--verbose] [--selftest]
```

The checker reads the printed numbers from the LaTeX source and the values from `results_cluster/`,
keeping the last written row of every configuration as `scripts/report.py` does. A printed number
passes when it equals the result rounded to the printed number of decimals.

## Licence

The code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE), and the aggregate
results in `results_cluster/`, `RESULTS.md` and `figures/data/` under
[Creative Commons Attribution-NonCommercial 4.0](LICENSE-results). Both allow research, teaching
and other noncommercial use, and neither allows commercial use. For commercial use, contact the
authors. The Roboto font in `figures/fonts/` keeps its own licence, the Apache License 2.0.

## Citation

If you use the code or the results, please cite the paper and this repository. The citation data of
the repository is in `CITATION.cff`; the reference of the paper will be added when it is published.
