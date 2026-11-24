"""Exporte FLORES+ (wolof et français) depuis Hugging Face vers l'arborescence
attendue par eval.mt_benchmark et eval.add_flores_fr :

    data/flores200_dataset/dev/wol_Latn.dev        data/flores200_dataset/devtest/wol_Latn.devtest
    data/flores200_dataset/dev/fra_Latn.dev        data/flores200_dataset/devtest/fra_Latn.devtest

Le jeu openlanguagedata/flores_plus est à accès restreint : créer un compte
Hugging Face, accepter ses conditions sur sa page, puis se connecter
(`hf auth login`) avant de lancer :

    python -m eval.export_flores --out data/flores200_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

LANGS = ["wol_Latn", "fra_Latn"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/flores200_dataset")
    ap.add_argument("--repo", default="openlanguagedata/flores_plus")
    a = ap.parse_args()
    from datasets import load_dataset

    for split in ("dev", "devtest"):
        d = Path(a.out) / split
        d.mkdir(parents=True, exist_ok=True)
        for lang in LANGS:
            ds = load_dataset(a.repo, lang, split=split)
            rows = sorted(ds, key=lambda r: int(r["id"])) if "id" in ds.column_names else list(ds)
            lines = [r["text"].replace("\n", " ").strip() for r in rows]
            (d / f"{lang}.{split}").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{split} {lang}: {len(lines)} phrases")


if __name__ == "__main__":
    main()
