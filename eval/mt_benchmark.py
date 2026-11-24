"""Benchmark traduction sur FLORES-200 (wol_Latn ↔ fra_Latn).

    python -m eval.mt_benchmark --flores-dir data/flores200_dataset --split devtest \
        --models facebook/nllb-200-distilled-600M facebook/nllb-200-1.3B facebook/nllb-200-3.3B

--flores-dir attend l'arborescence d'origine :
    <dir>/<split>/wol_Latn.<split>  et  <dir>/<split>/fra_Latn.<split>
(archive flores200_dataset publiée par Meta, ou export de openlanguagedata/flores_plus).

Métriques : chrF++ et spBLEU (sacrebleu, tokenisation flores200). Si sacrebleu
est absent, chrF++ est calculé par app.metrics (spBLEU non disponible).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.metrics import chrf_pp
from app.models import make_translator

from .common import run_metadata, save_json, write_csv


def scores(hyps: list[str], refs: list[str]) -> dict:
    try:
        import sacrebleu

        return {
            "chrfpp": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
            "spbleu": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score, 2),
        }
    except ImportError:
        return {"chrfpp": round(sum(chrf_pp(h, r) for h, r in zip(hyps, refs)) / len(refs), 2),
                "spbleu": None, "note": "sacrebleu absent : chrF++ moyen par phrase"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flores-dir", required=True)
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--models", nargs="+", default=["facebook/nllb-200-distilled-600M"])
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/mt")
    a = ap.parse_args()

    d = Path(a.flores_dir) / a.split
    wo = (d / f"wol_Latn.{a.split}").read_text(encoding="utf-8").splitlines()
    fr = (d / f"fra_Latn.{a.split}").read_text(encoding="utf-8").splitlines()
    assert len(wo) == len(fr), "Fichiers FLORES non alignés"
    if a.limit:
        wo, fr = wo[: a.limit], fr[: a.limit]

    out = Path(a.out)
    summary = {"meta": run_metadata(split=a.split, n=len(wo), beams=a.beams), "modeles": {}}
    for mid in a.models:
        print(f"\n=== {mid} ===")
        mt = make_translator(mid, "wol_Latn", "fra_Latn")  # « gemma4:e4b » → Ollama
        res = {}
        for direction, src, ref, rev in (("wol→fra", wo, fr, False), ("fra→wol", fr, wo, True)):
            t0 = time.perf_counter()
            hyps = []
            for i in range(0, len(src), a.batch):
                hyps += mt.translate_plain(src[i : i + a.batch], num_beams=a.beams, reverse=rev)
            dt = time.perf_counter() - t0
            s = scores(hyps, ref)
            s["secondes_par_phrase"] = round(dt / len(src), 3)
            res[direction] = s
            write_csv(out / mid.replace("/", "__").replace(":", "_") / f"{direction.replace('→', '-')}.csv",
                      [{"source": x, "reference": r, "hypothese": h} for x, r, h in zip(src, ref, hyps)])
            print(f"  {direction}: {s}")
        summary["modeles"][mid] = res
        del mt

    save_json(out / "summary.json", summary)
    md = ["| Modèle | Sens | chrF++ | spBLEU | s/phrase |", "|---|---|---|---|---|"]
    for m, r in summary["modeles"].items():
        for dirn, s in r.items():
            md.append(f"| {m} | {dirn} | {s['chrfpp']} | {s['spbleu']} | {s['secondes_par_phrase']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
