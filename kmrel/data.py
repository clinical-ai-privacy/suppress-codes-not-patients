"""Cohorts of MIMIC-IV v3.1 and eICU-CRD v2.0.

Release unit: the patient over all of his stays. Quasi-identifier A_p = (sex, ten-year age band),
known to the adversary. U_p = the patient's distinct diagnosis codes, version-prefixed
("9:4280", "10:I480"; no dot), so ICD-9 and ICD-10 never match below the root.

MIMIC: subject_id over all admissions; codes from
hosp/diagnoses_icd; sex = gender, band = anchor_age // 10 * 10; patients without codes or without
sex/age are outside the cohort.
eICU: uniquepid over all unit stays and hospitals; the ICD-9 part of
diagnosis.icd9code ("ICD9, ICD10"); ICD-10-only rows are dropped; sex Male/Female; age band with
"> 89" = 90; demographics = the first non-missing value in unitvisitnumber order; hospital = the
hospital of the patient's first diagnosis row.

The cohort is stored as CSR arrays: patient i has code ids cidx[indptr[i]:indptr[i+1]] (sorted,
distinct). Nothing here prints a patient-level value.

Data protection (PhysioNet DUA): this module runs only where the data are (the cluster or the
restricted folder). Callers print aggregates only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("KM_DATA", str(Path.home() / "km_data")))
MIMIC_HOSP = DATA / "mimic-iv-3.1" / "hosp"
EICU = DATA / "eicu-crd-2.0"


@dataclass
class Cohort:
    name: str
    pids: np.ndarray                 # original patient ids (sorted)
    strat: np.ndarray                # stratum index per patient
    strata: list                     # stratum labels (sex, band)
    indptr: np.ndarray               # CSR pointers, len = n + 1
    cidx: np.ndarray                 # code ids per patient, sorted within patient
    codes: list                      # code id -> code string
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.pids)

    def row(self, i):
        return self.cidx[self.indptr[i]:self.indptr[i + 1]]

    def lengths(self):
        return np.diff(self.indptr)

    def owner(self):
        """Patient index of every (patient, code) entry."""
        return np.repeat(np.arange(self.n), self.lengths())

    def stratum_members(self):
        """stratum -> sorted patient indices."""
        order = np.argsort(self.strat, kind="stable")
        cuts = np.searchsorted(self.strat[order], np.arange(len(self.strata) + 1))
        return [order[cuts[s]:cuts[s + 1]] for s in range(len(self.strata))]

    def subset(self, idx, name=None):
        """A sub-cohort treated as a whole population (strata kept, ids kept)."""
        idx = np.sort(np.asarray(idx))
        lens = self.lengths()[idx]
        indptr = np.concatenate([[0], np.cumsum(lens)])
        cidx = np.concatenate([self.row(i) for i in idx]) if len(idx) else np.zeros(0, np.int32)
        extra = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == self.n else v)
                 for k, v in self.extra.items()}
        return Cohort(name or self.name, self.pids[idx], self.strat[idx], self.strata,
                      indptr.astype(np.int64), cidx.astype(np.int32), self.codes, extra)

    def summary(self):
        lens = self.lengths()
        sizes = np.bincount(self.strat, minlength=len(self.strata))
        return dict(name=self.name, patients=int(self.n), strata=len(self.strata),
                    smallest_stratum=int(sizes.min()), codes_distinct=len(np.unique(self.cidx)),
                    pairs=int(len(self.cidx)), median_codes=float(np.median(lens)),
                    share_gt14=float((lens > 14).mean()), max_codes_ge=int(lens.max() >= 100))


def _build(name, pid_codes: pd.DataFrame, demo: pd.DataFrame, extra_cols=()):
    """pid_codes: columns pid, code (deduplicated or not); demo indexed by pid: gender, band."""
    pc = pid_codes.drop_duplicates(["pid", "code"])
    pc = pc[pc.pid.isin(demo.index)]
    pids = np.array(sorted(pc.pid.unique()))
    demo = demo.loc[pids]
    code_list = sorted(pc.code.unique())
    cid = {c: i for i, c in enumerate(code_list)}
    pos = pd.Series(np.arange(len(pids)), index=pids)
    pi = pos.loc[pc.pid.values].values
    ci = pc.code.map(cid).values
    order = np.lexsort((ci, pi))
    pi, ci = pi[order], ci[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(pi, minlength=len(pids)))])
    labels = sorted(set(zip(demo.gender, demo.band)))
    sidx = {l: i for i, l in enumerate(labels)}
    strat = np.array([sidx[(g, b)] for g, b in zip(demo.gender, demo.band)], dtype=np.int32)
    extra = {c: demo[c].values for c in extra_cols}
    return Cohort(name, pids, strat, labels, indptr.astype(np.int64), ci.astype(np.int32),
                  code_list, extra)


def load_mimic() -> Cohort:
    dx = pd.read_csv(MIMIC_HOSP / "diagnoses_icd.csv.gz", dtype={"icd_code": str},
                     usecols=["subject_id", "icd_code", "icd_version"])
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pts = pd.read_csv(MIMIC_HOSP / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    pts = pts[pts.gender.notna() & pts.anchor_age.notna()].copy()
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"
    demo = pts.set_index("subject_id")[["gender", "band"]]
    return _build("mimic", dx.rename(columns={"subject_id": "pid"})[["pid", "code"]], demo)


def _first_icd9(cell):
    if not isinstance(cell, str):
        return None
    code = cell.split(",")[0].strip()
    if not code:
        return None
    return code if code[:1].isdigit() or code[:1] in ("V", "E") else None


def _band(age):
    if isinstance(age, str):
        age = age.strip()
        if age.startswith(">"):
            age = "90"
        if not age:
            return None
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return None
    return f"{a // 10 * 10}s"


_EICU_CACHE = {}


def eicu_tables():
    """(pat, dxr): stay-level patient table with pid, and diagnosis rows
    with the ICD-9 part of icd9code, uniquepid, hospital and health-system stay."""
    if "t" in _EICU_CACHE:
        return _EICU_CACHE["t"]
    pat = pd.read_csv(EICU / "patient.csv.gz",
                      usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid",
                               "hospitalid", "gender", "age", "hospitaldischargestatus",
                               "unitvisitnumber", "hospitaldischargeoffset"],
                      dtype={"age": str})
    pat["band"] = pat["age"].map(_band)
    pat["gender"] = pat["gender"].where(pat["gender"].isin(["Male", "Female"]))
    dxr = pd.read_csv(EICU / "diagnosis.csv.gz",
                      usecols=["patientunitstayid", "icd9code", "diagnosisoffset"])
    dxr["raw"] = dxr["icd9code"].map(_first_icd9)
    dxr = dxr.dropna(subset=["raw"])
    dxr["code"] = "9:" + dxr["raw"].str.replace(".", "", regex=False)
    dxr = dxr.merge(pat[["patientunitstayid", "uniquepid", "hospitalid",
                         "patienthealthsystemstayid"]], on="patientunitstayid")
    pids = pd.Index(sorted(dxr["uniquepid"].unique()))
    pid_of = {p: i for i, p in enumerate(pids)}
    dxr["pid"] = dxr["uniquepid"].map(pid_of)
    pat["pid"] = pat["uniquepid"].map(pid_of)
    _EICU_CACHE["t"] = (pat, dxr)
    return pat, dxr


def load_eicu() -> Cohort:
    pat, dxr = eicu_tables()
    demo = (pat[pat.pid.notna()].assign(pid=lambda d: d.pid.astype(int))
            .sort_values(["pid", "unitvisitnumber"])
            .groupby("pid")[["gender", "band"]].first())
    demo = demo[demo.gender.notna() & demo.band.notna()]
    hosp = dxr.groupby("pid")["hospitalid"].first()
    nh = dxr.groupby("pid")["hospitalid"].nunique()
    demo["hospital"] = hosp.reindex(demo.index).values
    demo["n_hosp"] = nh.reindex(demo.index).values
    return _build("eicu", dxr[["pid", "code"]], demo, extra_cols=("hospital", "n_hosp"))


def load(name: str) -> Cohort:
    return {"mimic": load_mimic, "eicu": load_eicu}[name]()


# --------------------------------------------------------------------------- outcome tables ---
def mimic_admissions(coh: Cohort, sample=None):
    """One row per admission with diagnosis codes: subject_id, hadm_id, code ids (cohort ids),
    readmit30 (next admission 0-30 days after discharge; computed on all admissions),
    death (hospital_expire_flag). Restricted to the cohort (and to `sample` pids if given)."""
    adm = pd.read_csv(MIMIC_HOSP / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                               "hospital_expire_flag"], parse_dates=["admittime", "dischtime"])
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    adm["next_admit"] = adm.groupby("subject_id").admittime.shift(-1)
    gap = (adm.next_admit - adm.dischtime).dt.total_seconds() / 86400
    adm["readmit30"] = ((gap >= 0) & (gap <= 30)).astype(int)
    adm["death"] = adm.hospital_expire_flag.astype(int)
    dx = pd.read_csv(MIMIC_HOSP / "diagnoses_icd.csv.gz", dtype={"icd_code": str},
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    cid = {c: i for i, c in enumerate(coh.codes)}
    dx["cid"] = dx.code.map(cid)
    dx = dx.dropna(subset=["cid"])
    per = dx.groupby("hadm_id").cid.apply(lambda s: np.unique(s.astype(np.int32)))
    adm = adm[adm.hadm_id.isin(per.index) & adm.subject_id.isin(set(coh.pids))]
    if sample is not None:
        adm = adm[adm.subject_id.isin(set(sample))]
    adm = adm.reset_index(drop=True)
    adm["cids"] = per.reindex(adm.hadm_id).values
    pos = pd.Series(np.arange(coh.n), index=coh.pids)
    adm["pidx"] = pos.loc[adm.subject_id.values].values
    return adm[["subject_id", "pidx", "hadm_id", "cids", "readmit30", "death",
                "hospital_expire_flag"]]


def eicu_stays(coh: Cohort, first_day_only=False):
    """One row per health-system stay with known discharge status:
    pidx, codes (cohort ids), death = any unit stay 'Expired'. first_day_only keeps diagnosis
    rows with diagnosisoffset <= 1440 min (sensitivity analysis)."""
    pat, dxr = eicu_tables()
    d = dxr if not first_day_only else dxr[dxr.diagnosisoffset <= 1440]
    cid = {c: i for i, c in enumerate(coh.codes)}
    d = d.assign(cid=d.code.map(cid)).dropna(subset=["cid"])
    codes = d.groupby("patienthealthsystemstayid").cid.apply(lambda s: np.unique(s.astype(np.int32)))
    st = (pat.dropna(subset=["pid"]).groupby("patienthealthsystemstayid")
          .agg(pid=("pid", "first"),
               death=("hospitaldischargestatus", lambda s: int((s == "Expired").any())),
               known=("hospitaldischargestatus", lambda s: int(s.notna().any()))))
    st = st[st.known == 1].join(codes.rename("cids"), how="inner")
    st["pid"] = st.pid.astype(int)
    st = st[st.pid.isin(set(coh.pids))].reset_index()
    pos = pd.Series(np.arange(coh.n), index=coh.pids)
    st["pidx"] = pos.loc[st.pid.values].values
    return st[["patienthealthsystemstayid", "pid", "pidx", "cids", "death"]]
