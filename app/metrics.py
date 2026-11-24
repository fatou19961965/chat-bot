"""Métriques d'évaluation, implémentées sans dépendance lourde.

- Normalisation documentée (le cahier des charges exige qu'elle le soit :
  casse, ponctuation, chiffres, diacritiques).
- WER / CER par distance d'édition, avec détail S/D/I.
- chrF / chrF++ (Popović 2015/2017), version phrase, compatible avec la
  définition de sacrebleu (moyenne des précisions et rappels sur les ordres
  puis F-beta, beta = 2).

Le harnais d'évaluation (eval/) recoupe ces valeurs avec jiwer et sacrebleu
quand ils sont installés ; ces implémentations servent au suivi en ligne.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
# Ponctuation : tout caractère de catégorie Unicode P* ou S*, sauf l'apostrophe
# interne aux mots (ex. « j'ai », fréquent dans l'alternance wolof/français).
_APOSTROPHES = "'’ʼ`´"
_MULTISPACE = re.compile(r"\s+")
_HESITATIONS = {"euh", "heu", "hum", "hmm", "mm", "ee", "eee", "aa"}


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punct: bool = True,
    strip_diacritics: bool = False,
    drop_hesitations: bool = True,
    drop_tags: bool = True,
) -> str:
    """Normalisation appliquée à la référence ET à l'hypothèse avant WER/CER.

    Choix par défaut (voir docs/guide_transcription.md) :
      * NFC Unicode, minuscules ;
      * balises d'annotation [bruit], [rire], <fr>…</fr> retirées (texte gardé) ;
      * ponctuation supprimée, apostrophes unifiées ;
      * diacritiques CONSERVÉS : en wolof, ë/e, ó/o, à/a, ñ/n, ŋ/n changent le
        mot. `strip_diacritics=True` donne une mesure « tolérante » à reporter
        séparément ;
      * hésitations (euh, hum…) retirées ;
      * chiffres conservés tels quels (la convention CLAD écrit les nombres en
        toutes lettres : l'annotateur doit faire de même, voir le guide).
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    if drop_tags:
        t = re.sub(r"\[[^\]]*\]", " ", t)          # [bruit], [inaudible]
        t = re.sub(r"</?[a-zA-Z]+>", " ", t)        # <fr> … </fr>
    for a in _APOSTROPHES:
        t = t.replace(a, "'")
    if lowercase:
        t = t.lower()
    if strip_diacritics:
        t = "".join(
            c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"
        )
        t = t.replace("ŋ", "n")
        t = unicodedata.normalize("NFC", t)
    if remove_punct:
        out = []
        for i, c in enumerate(t):
            cat = unicodedata.category(c)
            if c == "'":
                prev_alpha = i > 0 and t[i - 1].isalpha()
                next_alpha = i + 1 < len(t) and t[i + 1].isalpha()
                out.append("'" if prev_alpha and next_alpha else " ")
            elif cat.startswith("P") or cat.startswith("S"):
                out.append(" ")
            else:
                out.append(c)
        t = "".join(out)
    words = _MULTISPACE.sub(" ", t).strip().split(" ")
    if drop_hesitations:
        words = [w for w in words if w and w not in _HESITATIONS]
    return " ".join(w for w in words if w)


# --------------------------------------------------------------------------
# Distance d'édition
# --------------------------------------------------------------------------
@dataclass
class EditStats:
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_len: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.ref_len == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.ref_len


def _edit_stats(ref: list, hyp: list) -> EditStats:
    n, m = len(ref), len(hyp)
    # d[i][j] = (coût, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            sub = prev[j - 1]
            dele = prev[j]
            ins = cur[j - 1]
            best = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                key=lambda x: x[0],
            )
            cur[j] = best
        prev = cur
    cost, s, d, ins_ = prev[m]
    hits = n - s - d
    return EditStats(s, d, ins_, hits, n)


def wer_stats(reference: str, hypothesis: str, normalize: bool = True, **norm_kw) -> EditStats:
    if normalize:
        reference, hypothesis = normalize_text(reference, **norm_kw), normalize_text(hypothesis, **norm_kw)
    return _edit_stats(reference.split(), hypothesis.split())


def wer(reference: str, hypothesis: str, normalize: bool = True, **norm_kw) -> float:
    return wer_stats(reference, hypothesis, normalize, **norm_kw).rate


def cer(reference: str, hypothesis: str, normalize: bool = True, **norm_kw) -> float:
    if normalize:
        reference, hypothesis = normalize_text(reference, **norm_kw), normalize_text(hypothesis, **norm_kw)
    return _edit_stats(list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))).rate


def corpus_wer(refs: list[str], hyps: list[str], **norm_kw) -> float:
    """WER corpus = somme des erreurs / somme des mots de référence (pas une moyenne de WER)."""
    err = tot = 0
    for r, h in zip(refs, hyps):
        st = wer_stats(r, h, **norm_kw)
        err += st.errors
        tot += st.ref_len
    return err / tot if tot else 0.0


def corpus_cer(refs: list[str], hyps: list[str], **norm_kw) -> float:
    err = tot = 0
    for r, h in zip(refs, hyps):
        r_n, h_n = normalize_text(r, **norm_kw), normalize_text(h, **norm_kw)
        st = _edit_stats(list(r_n.replace(" ", "")), list(h_n.replace(" ", "")))
        err += st.errors
        tot += st.ref_len
    return err / tot if tot else 0.0


# --------------------------------------------------------------------------
# chrF / chrF++
# --------------------------------------------------------------------------
def _char_ngrams(text: str, n: int) -> Counter:
    s = text.replace(" ", "")
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _word_ngrams(text: str, n: int) -> Counter:
    w = text.split()
    return Counter(tuple(w[i : i + n]) for i in range(len(w) - n + 1))


def chrf(
    hypothesis: str,
    reference: str,
    char_order: int = 6,
    word_order: int = 0,
    beta: float = 2.0,
) -> float:
    """chrF (word_order=0) ou chrF++ (word_order=2). Retourne un score 0..100."""
    hyp, ref = hypothesis.strip(), reference.strip()
    if not hyp and not ref:
        return 100.0
    if not hyp or not ref:
        return 0.0
    precs, recs = [], []
    for n in range(1, char_order + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        if not h or not r:
            continue
        match = sum((h & r).values())
        precs.append(match / sum(h.values()))
        recs.append(match / sum(r.values()))
    for n in range(1, word_order + 1):
        h, r = _word_ngrams(hyp, n), _word_ngrams(ref, n)
        if not h or not r:
            continue
        match = sum((h & r).values())
        precs.append(match / sum(h.values()))
        recs.append(match / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta**2
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def chrf_pp(hypothesis: str, reference: str) -> float:
    return chrf(hypothesis, reference, word_order=2)
