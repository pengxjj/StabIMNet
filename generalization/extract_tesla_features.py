#!/usr/bin/env python3
"""
Extract ALL TESLA features (Wells 2020 Cell mmc4=608 / mmc7=310) into clean CSVs.

Unified schema (one column per feature; NaN where a sheet lacks it):
  hla, peptide, label, pep_len, binding_affinity, tumor_abundance,
  binding_stability, frac_hydrophobic, agretopicity, foreignness,
  mutation_position, number_predicting

binding_affinity = NetMHCpan predicted IC50 (nM; LOWER = stronger binding).
Do NOT sign-flip here — that is done at eval time via -log10(aff).

Outputs (in data/TESLA/):
  tesla_608_features.csv   {hla, peptide, label, ...features}
  tesla_310_features.csv

Column map:
  mmc4 "master-bindings-selected":      mhc=3 pep=4 label=15
    pep_len=5 measured_aff=6 netmhc_aff=7 abund=8 stab=9 hydro=10 agre=11 fore=12
    mutpos=13 npred=14
  mmc7 "SM_Table_S7_PEPTIDE_VALIDATION_": pmhc=0 pep=3 label=12
    pep_len=4 pred_aff=5 netmhc_aff=6 abund=7 stab=8 agre=9 fore=10 mutpos=11
"""
import os
import csv
import openpyxl

HERE = os.path.dirname(os.path.realpath(__file__))
TESLA_DIR = os.path.join(HERE, "data", "TESLA")

MMC4 = os.path.join(TESLA_DIR, "1-s2.0-S0092867420311569-mmc4.xlsx")
MMC7 = os.path.join(TESLA_DIR, "1-s2.0-S0092867420311569-mmc7.xlsx")

FIELDS = ["hla", "peptide", "label", "pep_len", "binding_affinity",
          "tumor_abundance", "binding_stability", "frac_hydrophobic",
          "agretopicity", "foreignness", "mutation_position", "number_predicting"]


def norm_allele(a):
    a = str(a).strip().strip('"')
    if not a.startswith("HLA-"):
        a = "HLA-" + a
    return a


def tofloat(x):
    if x is None or x == "NA":
        return ""
    try:
        return float(x)
    except (ValueError, TypeError):
        return ""


def label_of(v):
    return 1 if v in (True, 1) else 0


def load_mmc4():
    wb = openpyxl.load_workbook(MMC4, read_only=True, data_only=True)
    ws = wb["master-bindings-selected"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[3] or not r[4]:
            continue
        rows.append({
            "hla": norm_allele(r[3]),
            "peptide": str(r[4]).strip(),
            "label": label_of(r[15]),
            "pep_len": tofloat(r[5]),
            "binding_affinity": tofloat(r[7]),   # NETMHC_PAN_BINDING_AFFINITY
            "tumor_abundance": tofloat(r[8]),
            "binding_stability": tofloat(r[9]),
            "frac_hydrophobic": tofloat(r[10]),
            "agretopicity": tofloat(r[11]),
            "foreignness": tofloat(r[12]),
            "mutation_position": tofloat(r[13]),
            "number_predicting": tofloat(r[14]),
        })
    wb.close()
    return rows


def load_mmc7():
    wb = openpyxl.load_workbook(MMC7, read_only=True, data_only=True)
    ws = wb["SM_Table_S7_PEPTIDE_VALIDATION_"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[0] or not r[3]:
            continue
        rows.append({
            "hla": norm_allele(str(r[0]).split("_")[0]),
            "peptide": str(r[3]).strip(),
            "label": label_of(r[12]),
            "pep_len": tofloat(r[4]),
            "binding_affinity": tofloat(r[6]),   # NETMHC_BINDING_AFFINITY
            "tumor_abundance": tofloat(r[7]),
            "binding_stability": tofloat(r[8]),
            "frac_hydrophobic": "",
            "agretopicity": tofloat(r[9]),
            "foreignness": tofloat(r[10]),
            "mutation_position": tofloat(r[11]),
            "number_predicting": "",
        })
    wb.close()
    return rows


def write_csv(name, rows):
    path = os.path.join(TESLA_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name}: n={len(rows)} pos={sum(r['label'] for r in rows)}")


def main():
    for name, rows in [("tesla_608_features.csv", load_mmc4()),
                       ("tesla_310_features.csv", load_mmc7())]:
        write_csv(name, rows)


if __name__ == "__main__":
    main()
