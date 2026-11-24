"""Benchmark ASR sur un manifeste (jeu interne + FLEURS).

    python -m eval.asr_benchmark --manifest data/test/manifest.csv \
        --models facebook/mms-1b-all facebook/mms-300m <whisper-wolof-communautaire> openai/whisper-large-v3 \
        --out results/asr

Produit, par modèle :
  * results/asr/<modele>/hypotheses.csv  (une ligne par énoncé : hyp, WER, CER, confiance, temps)
  * results/asr/summary.json / summary.md (WER/CER corpus, IC 95 % bootstrap,
    ventilation par condition, sexe, alternance codique, source ; RTF)

Remarque : facebook/mms-300m est un modèle pré-entraîné SANS tête CTC ; il
n'est utilisable qu'après fine-tuning (phase 3). Le script le signale au lieu
d'échouer silencieusement.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

from app import audio as A
from app.metrics import cer, corpus_cer, corpus_wer, wer
from app.models import ASR

from .common import bootstrap_ci, read_manifest, run_metadata, save_json, write_csv

GROUPS = ["condition", "gender", "code_switch", "source"]


def transcribe_file(asr: ASR, path: str) -> tuple[str, float, float]:
    wav = A.preprocess(A.decode_audio(Path(path).read_bytes()))
    dur = len(wav) / A.SR
    segs = A.vad_segments(wav, max_chunk_s=20) if dur > 25 else [A.Segment(0, dur, wav)]
    texts, confs, ws = [], [], []
    for s in segs:
        r = asr.transcribe(s.audio)
        if r.text:
            texts.append(r.text)
            confs.append(r.confidence)
            ws.append(s.duration)
    conf = sum(c * w for c, w in zip(confs, ws)) / sum(ws) if ws else 0.0
    return " ".join(texts), conf, dur


def jiwer_check(refs, hyps):
    try:
        import jiwer

        from app.metrics import normalize_text

        return jiwer.wer([normalize_text(r) for r in refs], [normalize_text(h) or "∅" for h in hyps])
    except ImportError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lang", default="wol")
    ap.add_argument("--out", default="results/asr")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = read_manifest(a.manifest)
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    summary = {"meta": run_metadata(manifest=a.manifest, n=len(rows)), "modeles": {}}

    for model_id in a.models:
        name = model_id.replace("/", "__")
        print(f"\n=== {model_id} ===")
        try:
            asr = ASR(model_id, a.lang)
        except Exception as e:
            print(f"  ! chargement impossible : {e}")
            summary["modeles"][model_id] = {"erreur": str(e)[:300]}
            continue
        per, refs, hyps = [], [], []
        t_audio = t_proc = 0.0
        for i, r in enumerate(rows, 1):
            t0 = time.perf_counter()
            try:
                hyp, conf, dur = transcribe_file(asr, r["audio_path"])
            except Exception as e:
                print(f"  ! {r['audio_path']} : {e}")
                continue
            dt = time.perf_counter() - t0
            t_audio += dur
            t_proc += dt
            refs.append(r["reference"])
            hyps.append(hyp)
            per.append({**r, "hypothese": hyp, "wer": round(wer(r["reference"], hyp), 4),
                        "cer": round(cer(r["reference"], hyp), 4),
                        "wer_sans_diacritiques": round(wer(r["reference"], hyp, strip_diacritics=True), 4),
                        "confiance": round(conf, 4), "duree_s": round(dur, 2), "temps_s": round(dt, 3)})
            if i % 20 == 0:
                print(f"  {i}/{len(rows)}  WER courant {corpus_wer(refs, hyps):.3f}")
        write_csv(out / name / "hypotheses.csv", per)

        by_group = {}
        for g in GROUPS:
            buckets = defaultdict(lambda: ([], []))
            for p in per:
                if p.get(g) not in (None, ""):
                    buckets[str(p[g])][0].append(p["reference"])
                    buckets[str(p[g])][1].append(p["hypothese"])
            if len(buckets) > 1 or (buckets and g == "source"):
                by_group[g] = {k: {"n": len(v[0]), "wer": round(corpus_wer(*v), 4)} for k, v in buckets.items()}

        lo, hi = bootstrap_ci(zip(refs, hyps), corpus_wer)
        res = {
            "n": len(per),
            "wer": round(corpus_wer(refs, hyps), 4),
            "wer_ic95": [round(lo, 4), round(hi, 4)],
            "cer": round(corpus_cer(refs, hyps), 4),
            "wer_sans_diacritiques": round(corpus_wer(refs, hyps, strip_diacritics=True), 4),
            "wer_jiwer_controle": jiwer_check(refs, hyps),
            "rtf": round(t_proc / t_audio, 3) if t_audio else None,
            "ventilation": by_group,
        }
        summary["modeles"][model_id] = res
        print(f"  WER {res['wer']:.3f} [{lo:.3f}–{hi:.3f}]  CER {res['cer']:.3f}  RTF {res['rtf']}")
        del asr

    save_json(out / "summary.json", summary)
    md = ["| Modèle | n | WER | IC 95 % | CER | WER sans diacritiques | RTF |", "|---|---|---|---|---|---|---|"]
    for m, r in summary["modeles"].items():
        if "erreur" in r:
            md.append(f"| {m} | — | échec : {r['erreur'][:60]} | | | | |")
        else:
            md.append(f"| {m} | {r['n']} | {r['wer']:.1%} | {r['wer_ic95'][0]:.1%}–{r['wer_ic95'][1]:.1%} "
                      f"| {r['cer']:.1%} | {r['wer_sans_diacritiques']:.1%} | {r['rtf']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
