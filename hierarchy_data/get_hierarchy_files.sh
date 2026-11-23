#!/bin/bash
# Downloads the two public CDC/NCHS files that kmrel/hierarchy.py reads, into this folder.
set -e
cd "$(dirname "$0")"
curl -L -o icd9cm_Dtab12.zip "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD9-CM/2011/Dtab12.zip"
curl -L -o icd10cm_table_index_2026.zip "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026/icd10cm-table%20and%20index-2026.zip"
unzip -o icd9cm_Dtab12.zip Dtab12.rtf
unzip -o icd10cm_table_index_2026.zip icd10cm-tabular-2026.xml
shasum -a 256 -c SHA256SUMS
