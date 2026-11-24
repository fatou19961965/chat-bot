"""Étalonnage du score de qualité estimé sur les énoncés corrigés.

    python scripts/calibrate_quality.py --db runtime/inferences.sqlite3

Cible : 1 - WER (transcription) et chrF++/100 (traduction) quand disponibles.
Ajuste par moindres carrés les poids (confiance ASR, confiance MT, aller-retour)
et affiche la corrélation avant/après. Les poids proposés se reportent dans
app/quality.py (W_ASR, W_MT, W_RT) après revue.
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runtime/inferences.sqlite3")
    ap.add_argument("--min", type=int, default=20)
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    rows = con.execute(
        "SELECT asr_conf, mt_conf, roundtrip, quality, wer, chrfpp FROM inferences "
        "WHERE (wer IS NOT NULL OR chrfpp IS NOT NULL) AND asr_conf IS NOT NULL"
    ).fetchall()
    if len(rows) < a.min:
        raise SystemExit(f"{len(rows)} énoncés corrigés : il en faut au moins {a.min} pour un étalonnage utile.")
    X, y, q = [], [], []
    for asr, mt, rt, qual, w, c in rows:
        targets = []
        if w is not None:
            targets.append(max(0.0, 1 - w))
        if c is not None:
            targets.append(c / 100)
        X.append([asr, mt, rt / 100])
        y.append(sum(targets) / len(targets))
        q.append(qual)
    X, y, q = np.array(X), np.array(y), np.array(q)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.array([0.4, 0.25, 0.35])
    pred = X @ w
    r_before = float(np.corrcoef(q, y)[0, 1])
    r_after = float(np.corrcoef(pred, y)[0, 1])
    print(f"n = {len(y)}")
    print(f"Corrélation actuelle score ↔ qualité mesurée : {r_before:.3f}")
    print(f"Corrélation après étalonnage               : {r_after:.3f}")
    print(f"Poids proposés : W_ASR={w[0]:.2f}  W_MT={w[1]:.2f}  W_RT={w[2]:.2f}")


if __name__ == "__main__":
    main()
