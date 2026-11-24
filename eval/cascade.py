"""Dégradation en cascade : qualité de traduction en fonction du WER d'entrée.

Résultat central du stage : le WER maximal tolérable pour que la chaîne reste
utilisable.

    python -m eval.cascade --hyps results/asr/facebook__mms-1b-all/hypotheses.csv \
        --threshold 35 --synthetic --out results/cascade

Deux sources de points :
  * RÉELLES : sorties ASR effectives (colonne `hypothese`) traduites par NLLB,
    comparées à `reference_fr` ; l'oracle est la traduction de la transcription
    de référence ;
  * SYNTHÉTIQUES (--synthetic) : bruit contrôlé injecté dans les références
    (substitutions/suppressions/insertions) à WER cible 0 → 80 %, pour obtenir
    une courbe dense même avec peu d'énoncés réels.

Hypothèse « ponctuation » : chaque point est calculé avec ET sans restauration
de ponctuation entre les deux étages (--no-punct-compare pour désactiver).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from app.metrics import chrf_pp, wer
from app.models import make_translator
from app.postprocess import restore_punctuation

from .common import read_manifest, run_metadata, save_json, write_csv

BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 10.0]


def corrupt(text: str, target_wer: float, vocab: list[str], rnd: random.Random) -> str:
    words = text.split()
    out = []
    for w in words:
        if rnd.random() < target_wer:
            op = rnd.choice(("sub", "sub", "del", "ins"))
            if op == "sub":
                out.append(rnd.choice(vocab))
            elif op == "ins":
                out += [w, rnd.choice(vocab)]
            # del : on n'ajoute rien
        else:
            out.append(w)
    return " ".join(out)


def translate(mt, texts: list[str], punct: bool, beams: int, batch: int = 16) -> list[str]:
    src = [restore_punctuation(t) if punct else t for t in texts]
    res = []
    for i in range(0, len(src), batch):
        res += mt.translate_plain(src[i : i + batch], num_beams=beams)
    return res


def bin_label(w: float) -> str:
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= w < hi:
            return f"{int(lo*100)}–{int(hi*100) if hi < 10 else '∞'} %"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True, help="hypotheses.csv issu de eval.asr_benchmark")
    ap.add_argument("--mt-model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=35.0,
                    help="chrF++ minimal jugé exploitable (à fixer avec l'évaluation humaine)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-punct-compare", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cascade")
    a = ap.parse_args()

    rows = [r for r in read_manifest(a.hyps) if r.get("reference_fr")]
    if not rows:
        raise SystemExit("Aucune ligne avec `reference_fr` : la cascade exige une traduction de référence.")
    mt = make_translator(a.mt_model, "wol_Latn", "fra_Latn")
    punct_modes = [True] if a.no_punct_compare else [True, False]
    refs_fr = [r["reference_fr"] for r in rows]
    refs_wo = [r["reference"] for r in rows]
    points = []

    for punct in punct_modes:
        oracle = translate(mt, refs_wo, punct, a.beams)
        real = translate(mt, [r["hypothese"] for r in rows], punct, a.beams)
        for r, o, h, rf in zip(rows, oracle, real, refs_fr):
            points.append({"source": "reel", "ponctuation": punct, "wer": float(r["wer"]),
                           "chrfpp": round(chrf_pp(h, rf), 2), "chrfpp_oracle": round(chrf_pp(o, rf), 2),
                           "audio_path": r["audio_path"], "traduction": h})

    if a.synthetic:
        rnd = random.Random(a.seed)
        vocab = sorted({w for t in refs_wo for w in t.lower().split()}) or ["a"]
        for target in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
            noisy = [corrupt(t, target, vocab, rnd) for t in refs_wo]
            for punct in punct_modes:
                hyps = translate(mt, noisy, punct, a.beams)
                for t, n, h, rf in zip(refs_wo, noisy, hyps, refs_fr):
                    points.append({"source": "synthetique", "ponctuation": punct,
                                   "wer": round(wer(t, n), 4), "chrfpp": round(chrf_pp(h, rf), 2)})

    out = Path(a.out)
    write_csv(out / "points.csv", points)

    # Agrégation par tranche de WER
    table = {}
    for p in points:
        k = (p["source"], p["ponctuation"], bin_label(p["wer"]))
        table.setdefault(k, []).append(p["chrfpp"])
    agg = [{"source": s, "ponctuation": pu, "tranche_wer": b, "n": len(v),
            "chrfpp_moyen": round(sum(v) / len(v), 2)} for (s, pu, b), v in sorted(table.items())]
    write_csv(out / "par_tranche.csv", agg)

    # WER maximal tolérable : plus haute tranche dont la moyenne reste ≥ seuil,
    # toutes les tranches inférieures l'étant aussi (courbe monotone attendue).
    def max_tolerable(src: str, punct: bool):
        best = None
        for lo, hi in zip(BINS, BINS[1:]):
            vals = [p["chrfpp"] for p in points if p["source"] == src and p["ponctuation"] == punct
                    and lo <= p["wer"] < hi]
            if len(vals) < 3:
                continue
            if sum(vals) / len(vals) >= a.threshold:
                best = hi if hi < 10 else lo
            else:
                break
        return best

    verdict = {
        f"{s}_ponct_{'oui' if pu else 'non'}": max_tolerable(s, pu)
        for s in ("reel", "synthetique") for pu in punct_modes
    }
    reel = [p for p in points if p["source"] == "reel" and p["ponctuation"]]
    summary = {
        "meta": run_metadata(mt_model=a.mt_model, seuil_chrfpp=a.threshold, n=len(rows)),
        "chrfpp_oracle_moyen": round(sum(p["chrfpp_oracle"] for p in reel) / len(reel), 2),
        "chrfpp_cascade_moyen": round(sum(p["chrfpp"] for p in reel) / len(reel), 2),
        "wer_max_tolerable": verdict,
    }
    if not a.no_punct_compare:
        nop = [p for p in points if p["source"] == "reel" and not p["ponctuation"]]
        summary["gain_ponctuation_chrfpp"] = round(
            summary["chrfpp_cascade_moyen"] - sum(p["chrfpp"] for p in nop) / len(nop), 2)
    save_json(out / "summary.json", summary)
    print(summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for src, mk in (("synthetique", "."), ("reel", "o")):
            for pu, col in ((True, "#2B3C8F"), (False, "#B23A48")):
                pts = [p for p in points if p["source"] == src and p["ponctuation"] == pu]
                if pts:
                    ax.scatter([p["wer"] * 100 for p in pts], [p["chrfpp"] for p in pts], s=14 if mk == "." else 28,
                               marker=mk, alpha=.45, color=col,
                               label=f"{src}, ponctuation {'restaurée' if pu else 'brute'}")
        ax.axhline(a.threshold, ls="--", color="#2F6B4F", label=f"seuil chrF++ = {a.threshold:g}")
        ax.set_xlabel("WER de la transcription en entrée (%)")
        ax.set_ylabel("chrF++ de la traduction")
        ax.set_title(f"Dégradation en cascade ASR → {a.mt_model}")
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "courbe_degradation.png", dpi=160)
        print(f"Courbe → {out / 'courbe_degradation.png'}")
    except ImportError:
        print("matplotlib absent : courbe non tracée (points.csv disponible).")


if __name__ == "__main__":
    main()
