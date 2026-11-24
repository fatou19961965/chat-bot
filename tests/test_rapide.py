"""Traducteur rapide (NLLB + CTranslate2) testé avec un faux tokenizer et un faux moteur."""
import math
from types import SimpleNamespace

from app.models import FastTranslator


class FakeTok:
    src_lang = "wol_Latn"

    def encode(self, t, truncation=True, max_length=256):
        return list(range(len(t.split()) + 2))

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def convert_tokens_to_ids(self, toks):
        return toks

    def decode(self, toks, skip_special_tokens=True):
        return "Je veux aller au marché." if toks and toks[0] == "fr" else "dama bëgg dem marse"


class FakeCT2:
    def __init__(self):
        self.calls = []

    def translate_batch(self, batch, target_prefix, **kw):
        # mêmes options que ctranslate2 4.9 : un argument inconnu doit échouer
        allowed = {"beam_size", "num_hypotheses", "return_scores", "length_penalty",
                   "max_decoding_length", "no_repeat_ngram_size"}
        assert set(kw) <= allowed, set(kw) - allowed
        self.calls.append((target_prefix, kw))
        tgt = target_prefix[0][0]
        hyp = [tgt, "fr" if tgt == "fra_Latn" else "wo"]
        return [SimpleNamespace(hypotheses=[hyp] * kw["num_hypotheses"], scores=[-0.5] * kw["num_hypotheses"])
                for _ in batch]


def _tr(roundtrip=False):
    t = FastTranslator.__new__(FastTranslator)
    t.model_id, t.src, t.tgt = "facebook/nllb-200-distilled-600M", "wol_Latn", "fra_Latn"
    t.roundtrip_enabled, t._cache, t.tok, t.ct2 = roundtrip, {}, FakeTok(), FakeCT2()
    return t


def test_traduction_rapide_un_seul_passage_et_cache():
    t = _tr()
    r = t.translate("dama bëgg dem marse")
    assert r.best.text == "Je veux aller au marché."
    assert math.isclose(r.best.confidence, math.exp(-0.25))
    assert t.translate("dama bëgg dem marse") is r
    assert len(t.ct2.calls) == 1
    assert t.ct2.calls[0][1]["beam_size"] == 1


def test_traduction_rapide_aller_retour():
    t = _tr(roundtrip=True)
    r = t.translate("dama bëgg dem marse")
    assert len(t.ct2.calls) == 2 and r.best.roundtrip > 90
