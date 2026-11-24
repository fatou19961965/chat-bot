"""Phase 3, axe (a) : fine-tuning de l'adaptateur wolof de MMS-1B.

On ne ré-entraîne que les couches d'adaptation (~2,5 M paramètres) et la tête
CTC : faisable sur un seul GPU de 16–24 Go, irréaliste sur CPU.

    python scripts/finetune_mms_adapter.py \
        --train data/kallaama/manifest_train.csv data/interne/manifest_train.csv \
        --dev data/interne/manifest_dev.csv --out models/mms-wol-ft --epochs 4

Puis mesurer le gain SUR LE MÊME HARNAIS :
    python -m eval.asr_benchmark --manifest data/test/manifest.csv \
        --models facebook/mms-1b-all models/mms-wol-ft
et servir le modèle : WVC_ASR_MODEL=models/mms-wol-ft

Garde-fou : ne JAMAIS mettre d'énoncés du jeu de test dans --train
(le script vérifie le recouvrement des chemins audio).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import audio as A  # noqa: E402
from app.metrics import corpus_wer, normalize_text  # noqa: E402
from eval.common import read_manifest  # noqa: E402


def load_rows(paths: list[str], vocab: set[str], max_s: float) -> list[dict]:
    rows = []
    for p in paths:
        for r in read_manifest(p):
            text = normalize_text(r["reference"])
            text = "".join(c for c in text if c in vocab or c == " ").strip()
            if not text:
                continue
            wav = A.preprocess(A.decode_audio(Path(r["audio_path"]).read_bytes()))
            if len(wav) / A.SR > max_s:
                continue
            rows.append({"audio": wav, "text": text, "path": r["audio_path"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", default="data/test/manifest.csv", help="pour le contrôle de fuite")
    ap.add_argument("--base", default="facebook/mms-1b-all")
    ap.add_argument("--lang", default="wol")
    ap.add_argument("--out", default="models/mms-wol-ft")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=20)
    a = ap.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Trainer, TrainingArguments, Wav2Vec2ForCTC
    from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

    processor = AutoProcessor.from_pretrained(a.base, target_lang=a.lang)
    processor.tokenizer.set_target_lang(a.lang)
    vocab = set(processor.tokenizer.get_vocab())
    vocab.discard(processor.tokenizer.word_delimiter_token)

    train = load_rows(a.train, vocab, a.max_seconds)
    dev = load_rows([a.dev], vocab, a.max_seconds)
    if Path(a.test).exists():
        leak = {r["path"] for r in train} & {r["audio_path"] for r in read_manifest(a.test)}
        if leak:
            raise SystemExit(f"Fuite : {len(leak)} fichiers du jeu de test présents dans l'entraînement.")
    print(f"train={len(train)}  dev={len(dev)}")

    model = Wav2Vec2ForCTC.from_pretrained(
        a.base, target_lang=a.lang, ignore_mismatched_sizes=True,
        layerdrop=0.0, ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer),
    )
    model.load_adapter(a.lang)
    model.freeze_base_model()
    for p in model._get_adapters().values():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînés : {n_train/1e6:.2f} M")

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            return {
                "input_values": processor(r["audio"], sampling_rate=A.SR).input_values[0],
                "labels": processor(text=r["text"]).input_ids,
            }

    @dataclass
    class Collator:
        def __call__(self, feats):
            batch = processor.pad([{"input_values": f["input_values"]} for f in feats],
                                  padding=True, return_tensors="pt")
            lab = processor.pad(labels=[{"input_ids": f["labels"]} for f in feats],
                                padding=True, return_tensors="pt")
            batch["labels"] = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
            return batch

    def compute_metrics(pred):
        ids = pred.predictions.argmax(-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(ids)
        refs = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": corpus_wer(refs, hyps)}

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=2, num_train_epochs=a.epochs, learning_rate=a.lr,
        warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch", logging_steps=25,
        fp16=torch.cuda.is_available(), group_by_length=False, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        report_to=[], seed=42,
    )
    trainer = Trainer(model=model, args=args, data_collator=Collator(), train_dataset=DS(train),
                      eval_dataset=DS(dev), compute_metrics=compute_metrics)
    trainer.train()

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    save_file(model._get_adapters(), os.path.join(a.out, WAV2VEC2_ADAPTER_SAFE_FILE.format(a.lang)),
              metadata={"format": "pt"})
    print(f"Modèle enregistré dans {a.out}")


if __name__ == "__main__":
    main()
