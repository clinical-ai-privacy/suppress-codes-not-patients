# Data needed to run the code

Nothing patient-level is in this package. Three inputs have to be obtained separately.

## 1. ICD hierarchy files (public, CDC / NCHS)

`kmrel/hierarchy.py` builds the official hierarchy (code, category, block, chapter, root) from
two public files, expected in `hierarchy_data/`:

| file | source | size |
|---|---|---|
| `Dtab12.rtf` (ICD-9-CM 2011 tabular list, diseases) | inside https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD9-CM/2011/Dtab12.zip | 0.8 MB zip, 13 MB rtf |
| `icd10cm-tabular-2026.xml` (ICD-10-CM FY2026 tabular list) | inside https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026/icd10cm-table%20and%20index-2026.zip | 19 MB zip, 10 MB xml |

Either run `hierarchy_data/get_hierarchy_files.sh` (curl + unzip, then a checksum check
against `SHA256SUMS`), or download the two zips by hand from the links above and unzip the named
file into `hierarchy_data/`. Both links were checked on 11 November 2026. The parent
directories are https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD9-CM/2011/ and
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026/ if a file name changes.

The checksums of the files we used:

```
e42a2cee1ec961352f2d8ea2bea95fe6b144c9168fa7d860c3f2f7789b78bd63  Dtab12.rtf
81f928efebb8f83d89e7265d6ba9fb72c9f853c80b92723021a9bfdde000d266  icd10cm-tabular-2026.xml
```

If the CDC publishes a newer ICD-10-CM year, the FY2026 file is the one the paper used; a newer
tabular list would change a handful of blocks and must not be mixed with our cached results.

## 2. MIMIC-IV 3.1 (PhysioNet, credentialed)

https://physionet.org/content/mimiciv/3.1/ , with your own credential and signed DUA. The code
reads, from `hosp/`: `patients.csv.gz`, `admissions.csv.gz`, `diagnoses_icd.csv.gz`.

## 3. eICU-CRD 2.0 (PhysioNet, credentialed)

https://physionet.org/content/eicu-crd/2.0/ . The code reads `patient.csv.gz` and
`diagnosis.csv.gz`.

## Where to put them

```
$KM_DATA/                      (default ~/km_data)
  mimic-iv-3.1/hosp/patients.csv.gz
  mimic-iv-3.1/hosp/admissions.csv.gz
  mimic-iv-3.1/hosp/diagnoses_icd.csv.gz
  eicu-crd-2.0/patient.csv.gz
  eicu-crd-2.0/diagnosis.csv.gz
```

Keep the PhysioNet files on the machine where they are allowed to be (your DUA), never in a
cloud-synced folder, and never upload them or any row of them to an online service, including
AI assistants. Only the aggregates written by `scripts/export_safe.py` should leave that machine.
