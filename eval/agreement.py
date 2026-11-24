"""Accords entre annotateurs — sans ces chiffres, les mesures ne valent rien.

1) Transcription (double annotation d'au moins 20 % du jeu de test) :
    python -m eval.agreement transcription --a annot_A.csv --b annot_B.csv
   Colonnes : audio_path, transcription. Mesure : WER/CER symétrique entre
   annotateurs (plancher de WER atteignable par un système), liste des
   énoncés litigieux à arbitrer.

2) Évaluation humaine de la traduction (100 phrases, 2 évaluateurs, 1–5) :
    python -m eval.agreement humaine --init --hyps results/mt/.../wol-fra.csv --n 100
        → crée eval_humaine_A.csv et eval_humaine_B.csv à remplir
    python -m eval.agreement humaine --a eval_humaine_A.csv --b eval_humaine_B.csv
   Mesure : moyennes d'adéquation et de fluidité, kappa de Cohen pondéré
   quadratique par critère.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from app.metrics import corpus_cer, corpus_wer, wer

from .common import save_json, write_csv


def _load(path: str, key: str = "audio_path") -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def transcription(a_path: str, b_path: str, out: Path, flag: float) -> None:
    A, B = _load(a_path), _load(b_path)
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit("Aucun énoncé commun aux deux annotations.")
    ta = [A[k]["transcription"] for k in keys]
    tb = [B[k]["transcription"] for k in keys]
    w_ab, w_ba = corpus_wer(ta, tb), corpus_wer(tb, ta)
    rows = []
    for k, x, y in zip(keys, ta, tb):
        d = (wer(x, y) + wer(y, x)) / 2
        rows.append({"audio_path": k, "annotateur_A": x, "annotateur_B": y,
                     "desaccord_wer": round(d, 4), "a_arbitrer": d > flag})
    write_csv(out / "accord_transcription.csv", rows)
    res = {
        "n_communs": len(keys),
        "wer_interannotateur": round((w_ab + w_ba) / 2, 4),
        "cer_interannotateur": round((corpus_cer(ta, tb) + corpus_cer(tb, ta)) / 2, 4),
        "a_arbitrer": sum(r["a_arbitrer"] for r in rows),
        "lecture": "Un système dont le WER est proche de cette valeur est au niveau de l'accord humain ; "
                   "une valeur élevée signale un guide de transcription à préciser.",
    }
    save_json(out / "accord_transcription.json", res)
    print(res)


def weighted_kappa(a: list[int], b: list[int], k_min: int = 1, k_max: int = 5) -> float | None:
    cats = list(range(k_min, k_max + 1))
    n = len(a)
    if n == 0:
        return None
    k = len(cats)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x - k_min][y - k_min] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j] / n
            den += w * ra[i] * cb[j] / (n * n)
    return None if den == 0 else round(1 - num / den, 4)


def human_init(hyps: str, n: int, out: Path, seed: int) -> None:
    rows = read_manifest_like(hyps)
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    for who in ("A", "B"):
        write_csv(out / f"eval_humaine_{who}.csv", [
            {"id": i, "source_wolof": r["source"], "traduction": r["hypothese"],
             "adequation_1a5": "", "fluidite_1a5": "", "commentaire": ""}
            for i, r in enumerate(rows)
        ])
    print(f"Fiches créées dans {out} (adéquation : le sens est-il rendu ? fluidité : le français est-il naturel ?)")


def read_manifest_like(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def human_score(a_path: str, b_path: str, out: Path) -> None:
    A, B = _load(a_path, "id"), _load(b_path, "id")
    keys = [k for k in A if k in B and A[k]["adequation_1a5"] and B[k]["adequation_1a5"]]
    res = {"n": len(keys)}
    for crit in ("adequation_1a5", "fluidite_1a5"):
        xa = [int(A[k][crit]) for k in keys]
        xb = [int(B[k][crit]) for k in keys]
        res[crit] = {
            "moyenne_A": round(sum(xa) / len(xa), 2) if xa else None,
            "moyenne_B": round(sum(xb) / len(xb), 2) if xb else None,
            "kappa_pondere": weighted_kappa(xa, xb),
        }
    save_json(out / "evaluation_humaine.json", res)
    print(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcription")
    t.add_argument("--a", required=True)
    t.add_argument("--b", required=True)
    t.add_argument("--flag", type=float, default=0.2)
    t.add_argument("--out", default="results/accord")
    h = sub.add_parser("humaine")
    h.add_argument("--init", action="store_true")
    h.add_argument("--hyps")
    h.add_argument("--n", type=int, default=100)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--a")
    h.add_argument("--b")
    h.add_argument("--out", default="results/humaine")
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "transcription":
        transcription(a.a, a.b, out, a.flag)
    elif a.init:
        human_init(a.hyps, a.n, out, a.seed)
    else:
        human_score(a.a, a.b, out)


if __name__ == "__main__":
    main()
