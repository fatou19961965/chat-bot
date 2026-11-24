"""Télécharge la partition wolof de FLEURS (google/fleurs, config wo_sn) et
écrit un manifeste compatible avec le harnais.

    python -m eval.prepare_fleurs --split test --out data/fleurs

FLEURS : parole LUE (textes FLoRes), donc plus facile que la parole spontanée ;
c'est la partition de comparabilité, pas de représentativité.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.audio import to_wav_bytes

from .common import write_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--out", default="data/fleurs")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from datasets import load_dataset  # dépendance d'évaluation

    ds = load_dataset("google/fleurs", "wo_sn", split=a.split, trust_remote_code=True)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        if a.limit and i >= a.limit:
            break
        wav = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        p = out / "audio" / f"fleurs_{a.split}_{ex['id']}_{i}.wav"
        p.write_bytes(to_wav_bytes(wav.astype("float32"), sr))
        rows.append({
            "audio_path": str(p),
            "reference": ex["transcription"],
            "reference_fr": "",
            "speaker_id": f"fleurs_{ex.get('speaker_id', '')}",
            "gender": {0: "M", 1: "F"}.get(ex.get("gender"), ""),
            "condition": "lu_studio",
            "code_switch": 0,
            "source": "fleurs",
        })
    write_csv(out / f"manifest_{a.split}.csv", rows)
    print(f"{len(rows)} énoncés → {out / f'manifest_{a.split}.csv'}")


if __name__ == "__main__":
    main()
