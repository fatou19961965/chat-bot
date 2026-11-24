"""Échoue (code 1) si le WER d'un modèle dépasse la référence enregistrée.

    python scripts/check_regression.py results/asr/summary.json \
        --baseline results_baseline/asr_summary.json --tolerance 0.01
"""
import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tolerance", type=float, default=0.01, help="hausse de WER absolue tolérée")
    a = ap.parse_args()
    cur = json.load(open(a.summary, encoding="utf-8"))["modeles"]
    base = json.load(open(a.baseline, encoding="utf-8"))["modeles"]
    bad = False
    for m, r in cur.items():
        if m in base and "wer" in r and "wer" in base[m]:
            delta = r["wer"] - base[m]["wer"]
            flag = delta > a.tolerance
            bad |= flag
            print(f"{'RÉGRESSION' if flag else 'ok'}  {m}: WER {base[m]['wer']:.3f} → {r['wer']:.3f} ({delta:+.3f})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
