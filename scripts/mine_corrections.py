"""Extrait des corrections utilisateurs les substitutions de mots récurrentes
(variante reconnue → forme corrigée) et les propose pour le lexique.

    python scripts/mine_corrections.py --db runtime/inferences.sqlite3 --min-count 3

Écrit data/lexique_candidats.tsv. Les candidats ne sont PAS appliqués
automatiquement : un annotateur les valide puis les copie dans
data/lexique_corrections.tsv.
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from app.metrics import normalize_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runtime/inferences.sqlite3")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--out", default="data/lexique_candidats.tsv")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    rows = con.execute(
        "SELECT transcript, ref_transcript FROM inferences WHERE ref_transcript IS NOT NULL"
    ).fetchall()
    subs: Counter = Counter()
    for hyp, ref in rows:
        h, r = normalize_text(hyp or "").split(), normalize_text(ref or "").split()
        for op, i1, i2, j1, j2 in SequenceMatcher(a=h, b=r, autojunk=False).get_opcodes():
            if op == "replace" and (i2 - i1) == (j2 - j1):
                for x, y in zip(h[i1:i2], r[j1:j2]):
                    subs[(x, y)] += 1
    keep = [(x, y, n) for (x, y), n in subs.most_common() if n >= a.min_count]
    Path(a.out).write_text(
        "# variante\tforme_corrigee\toccurrences — À VALIDER avant copie dans lexique_corrections.tsv\n"
        + "".join(f"{x}\t{y}\t{n}\n" for x, y, n in keep), encoding="utf-8")
    print(f"{len(rows)} énoncés corrigés, {len(keep)} substitutions récurrentes → {a.out}")


if __name__ == "__main__":
    main()
