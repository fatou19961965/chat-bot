"""Chaîne complète : audio wolof → transcription → post-traitement →
traduction (n-best + reclassement) → estimation de qualité → voix française.

Chaque étage est chronométré ; tout est journalisé pour l'évaluation.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass

from . import audio as A
from .config import settings
from .memory import get_memory
from .models import get_asr, get_translator, get_tts
from .postprocess import apply_lexicon, code_switch_ratio, restore_punctuation
from .quality import estimate
from .storage import Store

log = logging.getLogger("wvc.pipeline")


class PipelineError(ValueError):
    pass


# En dessous de ce seuil de cohérence aller-retour (chrF 0..100), la traduction
# est jugée incertaine : on l'affiche, mais la voix demande de répéter plutôt
# que de lire une traduction probablement fausse.
ROUNDTRIP_MIN = settings.roundtrip_min
MSG_INCERTAIN = "Je ne suis pas sûr d'avoir bien compris. Pouvez-vous répéter plus lentement ?"
MSG_BRUIT = ("Trop de bruit : aucune parole claire détectée. Rapprochez-vous du micro "
             "ou éloignez-vous du bruit, puis réessayez.")


@dataclass
class Options:
    tts: bool = True
    punctuation: bool = settings.punctuation_restore
    n_best: int = settings.n_best


def _weighted(values: list[float], weights: list[float]) -> float:
    tw = sum(weights)
    return float(sum(v * w for v, w in zip(values, weights)) / tw) if tw else 0.0


def _translate_segment(mt, text: str, n_best: int):
    """Mémoire de traduction d'abord, modèle de traduction (Gemma ou NLLB) sinon.
    Retourne (traduction, confiance, aller_retour, candidats, source)."""
    m = get_memory().lookup(text)
    if m is not None:
        src = "memoire_exacte" if m.exacte else "memoire_approchee"
        cands = [{"texte": m.francais, "confiance": m.similarite, "aller_retour": 100.0 * m.similarite,
                  "score": m.similarite, "source": src, "entree_memoire": m.wolof}]
        return m.francais, m.similarite, 100.0 * m.similarite, cands, src
    backend = getattr(mt, "backend", "nllb")
    res = mt.translate(text, n_best=n_best, num_beams=settings.num_beams, alpha=settings.rerank_alpha)
    cands = [{"texte": c.text, "confiance": round(c.confidence, 4), "aller_retour": round(c.roundtrip, 2),
              "score": round(c.score, 4), "source": backend} for c in res.candidates]
    return res.best.text, res.best.confidence, res.best.roundtrip, cands, backend


def _memory_summary(sources: list[str]) -> dict:
    n = len(sources)
    hits = sum(s.startswith("memoire") for s in sources)
    return {"segments": n, "depuis_memoire": hits,
            "approchee": any(s == "memoire_approchee" for s in sources)}


def run(data: bytes, store: Store, opts: Options | None = None, filename: str = "audio.webm") -> dict:
    opts = opts or Options()
    t: dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. Décodage et contrôles ---------------------------------------------
    try:
        wav = A.decode_audio(data)
    except A.AudioError as e:
        raise PipelineError(str(e)) from e
    duration = len(wav) / A.SR
    if duration > settings.max_audio_s:
        raise PipelineError(
            f"Audio trop long ({duration:.0f} s) : la limite est de {settings.max_audio_s:.0f} s."
        )
    if duration < 0.3:
        raise PipelineError("Audio trop court : maintenez le bouton pendant que vous parlez.")
    wav = A.preprocess(wav, denoise_strength=settings.denoise_strength, denoise_floor=settings.denoise_floor)
    segments = A.vad_segments(
        wav, max_chunk_s=settings.vad_max_chunk_s, min_silence_s=settings.vad_min_silence_s,
        min_snr_db=settings.min_snr_db, min_speech_s=settings.vad_min_speech_s, pad_s=settings.vad_pad_s,
    )
    if not segments:
        raise PipelineError(MSG_BRUIT)
    t["decodage_vad"] = time.perf_counter() - t0

    # 2. ASR par segment ----------------------------------------------------
    t1 = time.perf_counter()
    asr = get_asr()
    raw_parts, confs, weights, rejected = [], [], [], 0
    doubtful: list[tuple] = []   # segments écartés, gardés en réserve
    min_conf = getattr(asr, "min_conf", settings.asr_min_conf)
    for seg in segments:
        r = asr.transcribe(seg.audio, A.SR)
        # Segment trop incertain ou quasi vide : c'est du bruit que le modèle a
        # « transcrit ». On l'ignore au lieu de traduire n'importe quoi.
        if r.text and (r.confidence < min_conf or len(r.text.replace(" ", "")) < 2):
            log.info("Segment ignoré (bruit probable) : conf=%.2f « %s »", r.confidence, r.text[:40])
            rejected += 1
            if len(r.text.split()) >= 2 and r.confidence >= min_conf / 2:
                doubtful.append((r, seg))
            continue
        if r.text:
            raw_parts.append(r.text)
            confs.append(r.confidence)
            weights.append(seg.duration)
    t["asr"] = time.perf_counter() - t1
    if not raw_parts and doubtful:
        # Tout a été jugé douteux, mais il y a de vrais mots : on les garde,
        # et le résultat sera signalé « incertain » plutôt que refusé.
        for r, seg in doubtful:
            raw_parts.append(r.text)
            confs.append(r.confidence)
            weights.append(seg.duration)
        rejected -= len(doubtful)
    if not raw_parts:
        raise PipelineError(MSG_BRUIT if rejected else
            "Aucune parole reconnue : parlez plus près du micro, puis appuyez à nouveau pour envoyer."
        )
    asr_conf = _weighted(confs, weights)

    # 3. Post-traitement ----------------------------------------------------
    post_parts, n_lex = [], 0
    for p in raw_parts:
        p2, n = apply_lexicon(p)
        n_lex += n
        post_parts.append(restore_punctuation(p2) if opts.punctuation else p2)
    transcript = " ".join(raw_parts)
    transcript_post = " ".join(post_parts)

    # 4. Traduction par segment, n-best + reclassement aller-retour -------
    t2 = time.perf_counter()
    mt = get_translator()
    fr_parts, mt_confs, rts, cands_log, sources = [], [], [], [], []
    mt_parts = post_parts
    if getattr(mt, "backend", "") == "gemma" and settings.gemma_merge_segments and len(post_parts) > 1:
        # Un seul appel à Gemma pour tout l'enregistrement : l'invite (consignes +
        # exemples) n'est lue qu'une fois, et le modèle voit la phrase entière.
        mt_parts = [" ".join(post_parts)]
    for p in mt_parts:
        try:
            fr, conf, rt, cands, src = _translate_segment(mt, p, opts.n_best)
        except Exception as e:  # message clair au lieu d'une « erreur 500 »
            log.exception("Traduction en échec")
            raise PipelineError(f"Erreur du traducteur ({type(e).__name__}). Voir le terminal.") from e
        fr_parts.append(fr)
        mt_confs.append(conf)
        rts.append(rt)
        cands_log.append(cands)
        sources.append(src)
    t["traduction"] = time.perf_counter() - t2
    translation = " ".join(x for x in fr_parts if x)
    w_txt = [max(1, len(p.split())) for p in mt_parts]
    mt_conf = _weighted(mt_confs, w_txt)
    roundtrip = _weighted(rts, w_txt)

    # 5. Qualité estimée ---------------------------------------------------
    cs = code_switch_ratio(transcript)
    q = estimate(asr_conf, mt_conf, roundtrip, cs, n_words=len(transcript.split()))
    rt_on = getattr(mt, "roundtrip_enabled", True)
    incertain = q.level == "faible" or (rt_on and roundtrip < ROUNDTRIP_MIN)

    # 6. Synthèse vocale française ----------------------------------------
    audio_b64 = None
    if opts.tts and translation:
        t3 = time.perf_counter()
        try:
            out, sr = get_tts().synthesize(translation)
            audio_b64 = base64.b64encode(A.to_wav_bytes(out, sr)).decode()
        except Exception as e:  # la voix est un confort : on ne fait pas échouer la chaîne
            log.warning("TTS en échec : %s", e)
        t["synthese"] = time.perf_counter() - t3

    # 7. Réponse conversationnelle optionnelle ----------------------------
    reply = None
    if settings.llm_enabled and translation:
        from .assistant import reply as llm_reply

        t4 = time.perf_counter()
        reply = llm_reply(translation)
        t["assistant"] = time.perf_counter() - t4

    t["total"] = time.perf_counter() - t0
    log.warning(
        "TEMPS  audio %.1f s | reconnaissance %.1f s | traduction %.1f s | voix %.1f s | TOTAL %.1f s  (%s + %s)",
        duration, t.get("asr", 0), t.get("traduction", 0), t.get("synthese", 0), t["total"],
        getattr(get_asr(), "model_id", "?"), getattr(mt, "model_id", "?"),
    )

    # 8. Journalisation ----------------------------------------------------
    iid = store.new_id()
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "webm")[:5]
    store.insert({
        "id": iid,
        "audio_path": store.save_audio(iid, data, ext),
        "duration_s": round(duration, 2),
        "n_segments": len(segments),
        "transcript": transcript,
        "transcript_post": transcript_post,
        "translation": translation,
        "candidates": cands_log,
        "asr_conf": q.asr_confidence,
        "mt_conf": q.mt_confidence,
        "roundtrip": q.roundtrip_chrf,
        "code_switch": q.code_switch,
        "quality": q.score,
        "level": q.level,
        "timings": {k: round(v, 3) for k, v in t.items()},
        "models": {
            "asr": getattr(asr, "model_id", "?"),
            "mt": getattr(mt, "model_id", "?"),
            "lexique_remplacements": n_lex,
            "memoire": _memory_summary(sources),
            "segments_ignores_bruit": rejected,
        },
    })

    return {
        "id": iid,
        "duree_s": round(duration, 2),
        "segments": [
            {"debut": round(s.start, 2), "fin": round(s.end, 2)} for s in segments
        ],
        "segments_ignores_bruit": rejected,
        "transcription": transcript,
        "transcription_post": transcript_post,
        "traduction": translation,
        "candidats": cands_log,
        "memoire": _memory_summary(sources),
        "qualite": q.to_dict(),
        "incertain": incertain,
        "message_incertain": MSG_INCERTAIN if incertain else None,
        "reponse": reply,
        "audio_fr_wav_b64": audio_b64,
        "temps_s": {k: round(v, 3) for k, v in t.items()},
    }


def translate_text(text: str, store: Store | None = None, n_best: int | None = None) -> dict:
    """Traduction d'un texte wolof saisi au clavier. Même format de réponse que
    `run` (sans audio) et même journalisation : l'énoncé peut être corrigé."""
    t0 = time.perf_counter()
    p, n_lex = apply_lexicon(text.strip())
    post = restore_punctuation(p) if settings.punctuation_restore else p
    mt = get_translator()
    fr, conf, rt, seg_cands, src = _translate_segment(mt, post, n_best or settings.n_best)
    cands = [seg_cands]
    cs = code_switch_ratio(text)
    # Pas d'étage ASR : sa confiance est neutre (1.0), le score ne reflète que la traduction.
    q = estimate(1.0, conf, rt, cs, n_words=len(text.split()))
    incertain = q.level == "faible" or (getattr(mt, "roundtrip_enabled", True) and rt < ROUNDTRIP_MIN)
    timings = {"traduction": round(time.perf_counter() - t0, 3)}
    iid = None
    if store is not None:
        iid = store.new_id()
        store.insert({
            "id": iid, "audio_path": None, "duration_s": 0, "n_segments": 0,
            "transcript": text.strip(), "transcript_post": post, "translation": fr,
            "candidates": cands, "asr_conf": None, "mt_conf": q.mt_confidence,
            "roundtrip": q.roundtrip_chrf, "code_switch": cs, "quality": q.score, "level": q.level,
            "timings": timings,
            "models": {"asr": "saisie clavier", "mt": getattr(mt, "model_id", "?"),
                       "lexique_remplacements": n_lex, "memoire": _memory_summary([src])},
        })
    return {
        "id": iid,
        "transcription": text.strip(),
        "transcription_post": post,
        "traduction": fr,
        "candidats": cands,
        "memoire": _memory_summary([src]),
        "qualite": q.to_dict(),
        "incertain": incertain,
        "message_incertain": MSG_INCERTAIN if incertain else None,
        "reponse": None,
        "audio_fr_wav_b64": None,
        "saisie": True,
        "temps_s": timings,
    }


__all__ = ["run", "translate_text", "Options", "PipelineError"]
