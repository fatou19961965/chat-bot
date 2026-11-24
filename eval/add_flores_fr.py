"""Complète la colonne `reference_fr` d'un manifeste FLEURS.

Les phrases de FLEURS sont des phrases de FLORES lues à voix haute : on
retrouve donc la traduction française de chaque énoncé en alignant sa
transcription wolof sur FLORES (normalisation + correspondance approchée).

    python -m eval.add_flores_fr --manifest data/fleurs/manifest_test.csv \
        --flores-dir data/flores200_dataset --out data/fleurs/manifest_test_fr.csv
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

from app.metrics import normalize_text

from .common import read_manifest, write_csv


def _k(t: str) -> str:
    return normalize_text(t, strip_diacritics=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--flores-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-sim", type=float, default=0.9)
    a = ap.parse_args()

    table: dict[str, str] = {}
    for split in ("dev", "devtest"):
        d = Path(a.flores_dir) / split
        wo, fr = d / f"wol_Latn.{split}", d / f"fra_Latn.{split}"
        if wo.exists() and fr.exists():
            for w, f in zip(wo.read_text(encoding="utf-8").splitlines(), fr.read_text(encoding="utf-8").splitlines()):
                table[_k(w)] = f
    if not table:
        raise SystemExit("FLORES introuvable : lancez d'abord python -m eval.export_flores")

    keys = list(table)
    rows = read_manifest(a.manifest)
    exact = approx = 0
    for r in rows:
        k = _k(r["reference"])
        if k in table:
            r["reference_fr"] = table[k]
            exact += 1
            continue
        best, sim = None, 0.0
        for fk in keys:
            if abs(len(fk) - len(k)) > 0.2 * max(len(k), 1):
                continue
            s = SequenceMatcher(None, k, fk).ratio()
            if s > sim:
                best, sim = fk, s
        if best and sim >= a.min_sim:
            r["reference_fr"] = table[best]
            approx += 1
    write_csv(a.out, rows)
    n = len(rows)
    print(f"{n} énoncés : {exact} alignés exactement, {approx} approximativement, "
          f"{n - exact - approx} sans traduction → {a.out}")


if __name__ == "__main__":
    main()
