"""Utilitaires partagés par le harnais d'évaluation."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REQUIRED = ["audio_path", "reference"]


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in REQUIRED if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"Colonnes manquantes dans {path} : {missing}")
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_metadata(**extra) -> dict:
    """Tout ce qu'il faut pour qu'un tiers retrouve les chiffres."""
    def _git():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    versions = {}
    for mod in ("torch", "transformers", "jiwer", "sacrebleu", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": _git(),
        "python": sys.version.split()[0],
        "plateforme": platform.platform(),
        "versions": versions,
        **extra,
    }


def save_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def bootstrap_ci(values_ref_hyp, metric_fn, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Intervalle de confiance à 95 % par ré-échantillonnage des énoncés."""
    import random

    rnd = random.Random(seed)
    data = list(values_ref_hyp)
    if len(data) < 2:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n):
        s = [data[rnd.randrange(len(data))] for _ in data]
        stats.append(metric_fn([a for a, _ in s], [b for _, b in s]))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]
