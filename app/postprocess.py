"""Post-traitement entre ASR et traduction (axe (b) du sujet de stage).

1. Lexique de corrections : variantes orthographiques fréquentes → forme CLAD.
   Le fichier data/lexique_corrections.tsv s'enrichit à partir des corrections
   saisies par les utilisateurs (voir scripts/mine_corrections.py).
2. Restauration heuristique de la ponctuation et de la casse : les sorties CTC
   sont en minuscules sans ponctuation, alors que NLLB a été entraîné sur des
   phrases ponctuées. L'hypothèse « ponctuation » du sujet se mesure en
   activant/désactivant WVC_PUNCT_RESTORE dans eval/cascade.py.

Cette version est volontairement simple et transparente ; elle sert de base
de comparaison à un éventuel modèle appris.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "lexique_corrections.tsv"

# Mots interrogatifs wolof (orthographe CLAD). Présence en tête de segment →
# phrase interrogative.
QUESTION_WORDS = {
    "ndax", "ana", "fan", "fu", "kan", "ku", "kañ", "lan", "lu", "luy",
    "naka", "nan", "ñaata", "ban", "bu", "lutax", "lu tax", "ndaxam",
}
# Petit lexique de mots français fréquents dans l'alternance codique urbaine.
FRENCH_MARKERS = {
    "mais", "donc", "parce", "que", "alors", "bon", "voilà", "normalement",
    "déjà", "vraiment", "problème", "compte", "banque", "argent", "téléphone",
    "rendez-vous", "d'accord", "c'est", "il", "elle", "je", "vous", "nous",
    "le", "la", "les", "des", "une", "un", "est", "pas", "avec", "pour",
}


@lru_cache(maxsize=1)
def load_lexicon(path: str = str(LEXICON_PATH)) -> dict[str, str]:
    p = Path(path)
    lex: dict[str, str] = {}
    if not p.exists():
        return lex
    with p.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or row[0].startswith("#") or len(row) < 2:
                continue
            lex[row[0].strip().lower()] = row[1].strip()
    return lex


def apply_lexicon(text: str, lexicon: dict[str, str] | None = None) -> tuple[str, int]:
    """Remplace les variantes connues, mot à mot. Retourne (texte, nb de remplacements)."""
    lexicon = load_lexicon() if lexicon is None else lexicon
    if not lexicon or not text:
        return text, 0
    n = 0
    out = []
    for w in text.split():
        rep = lexicon.get(w.lower())
        if rep is not None and rep != w:
            n += 1
            out.append(rep)
        else:
            out.append(w)
    return " ".join(out), n


def is_question(segment: str) -> bool:
    words = segment.lower().split()
    if not words:
        return False
    head = words[:2]
    return any(w in QUESTION_WORDS for w in head) or " ".join(head) in QUESTION_WORDS


def restore_punctuation(segment: str) -> str:
    """Majuscule initiale + ponctuation finale selon la modalité détectée."""
    s = re.sub(r"\s+", " ", segment).strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] in ".?!":
        return s
    return s + ("?" if is_question(s) else ".")


def code_switch_ratio(text: str) -> float:
    """Part (0..1) de mots appartenant au lexique de marqueurs français.
    Indicateur grossier, utilisé pour qualifier les échecs (limite structurelle
    connue des modèles monolingues)."""
    words = [w.strip(".,?!").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return sum(w in FRENCH_MARKERS for w in words) / len(words)
