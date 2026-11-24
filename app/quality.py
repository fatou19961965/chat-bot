"""Suivi de la qualité, en deux régimes.

1. SANS référence (en direct, à chaque énoncé) : score ESTIMÉ, combinaison de
   signaux internes. Ce n'est pas un WER : c'est un indicateur de risque qui
   permet d'alerter l'utilisateur (« répétez, je ne suis pas sûr ») et de
   prioriser les énoncés à faire corriger.

2. AVEC référence (après correction par l'utilisateur, ou sur le jeu de test) :
   WER, CER, chrF++ réels. Ces mesures servent aussi à ÉTALONNER le score
   estimé : scripts/calibrate_quality.py ajuste les poids par régression sur
   les énoncés corrigés.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import settings
from .metrics import cer, chrf_pp, normalize_text, wer

# Poids par défaut (à recalibrer sur données réelles).
W_ASR, W_MT, W_RT = 0.40, 0.25, 0.35


@dataclass
class QualityEstimate:
    asr_confidence: float    # 0..1
    mt_confidence: float     # 0..1
    roundtrip_chrf: float    # 0..100
    code_switch: float       # 0..1
    score: float             # 0..1
    level: str               # bon | moyen | faible
    advice: str

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def estimate(asr_conf: float, mt_conf: float, roundtrip: float, code_switch: float = 0.0,
             n_words: int = 0) -> QualityEstimate:
    score = W_ASR * asr_conf + W_MT * mt_conf + W_RT * (roundtrip / 100.0)
    # Pénalités : énoncé quasi vide, forte alternance codique (limite connue
    # des modèles monolingues).
    if n_words <= 1:
        score *= 0.85
    score *= 1.0 - 0.3 * min(1.0, code_switch * 2)
    score = max(0.0, min(1.0, score))

    if score >= settings.quality_good:
        level, advice = "bon", "Traduction fiable."
    elif score >= settings.quality_poor:
        level, advice = "moyen", "Traduction probablement correcte ; vérifiez le sens."
    else:
        weakest = min(
            (asr_conf, "La reconnaissance est incertaine : parlez plus près du micro ou plus lentement."),
            (mt_conf, "La traduction est incertaine : reformulez en phrases plus courtes."),
            (roundtrip / 100.0, "Le sens a pu être altéré : reformulez."),
            key=lambda x: x[0],
        )[1]
        level, advice = "faible", weakest
    return QualityEstimate(asr_conf, mt_conf, roundtrip, code_switch, score, level, advice)


def measured(
    hyp_transcript: str,
    hyp_translation: str,
    ref_transcript: str | None = None,
    ref_translation: str | None = None,
) -> dict:
    """Métriques réelles, quand une référence est disponible."""
    out: dict = {}
    if ref_transcript:
        out["wer"] = round(wer(ref_transcript, hyp_transcript), 4)
        out["cer"] = round(cer(ref_transcript, hyp_transcript), 4)
        out["wer_sans_diacritiques"] = round(
            wer(ref_transcript, hyp_transcript, strip_diacritics=True), 4
        )
    if ref_translation:
        out["chrfpp"] = round(
            chrf_pp(normalize_text(hyp_translation, lowercase=False, remove_punct=False),
                    normalize_text(ref_translation, lowercase=False, remove_punct=False)),
            2,
        )
    return out
