"""Configuration centralisée (variables d'environnement, préfixe WVC_)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on", "oui"}


def _list(name: str) -> list[str]:
    v = os.getenv(name, "")
    return [x.strip() for x in v.split(",") if x.strip()]


_OLLAMA_URL = os.getenv("WVC_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
_GEMMA_MODEL = os.getenv("WVC_GEMMA_MODEL", "gemma4:e4b")


@dataclass(frozen=True)
class Settings:
    # --- Modèles -------------------------------------------------------------
    asr_model: str = os.getenv("WVC_ASR_MODEL", "facebook/mms-1b-all")
    asr_lang: str = os.getenv("WVC_ASR_LANG", "wol")
    # Moteur de traduction : "rapide" (NLLB int8, CTranslate2), "gemma" (Ollama) ou "nllb".
    mt_backend: str = os.getenv("WVC_MT_BACKEND", "gemma").strip().lower()
    mt_model: str = os.getenv("WVC_MT_MODEL", "facebook/nllb-200-distilled-600M")
    rapide_model: str = os.getenv("WVC_RAPIDE_MODELE", "facebook/nllb-200-distilled-600M")
    rapide_roundtrip: bool = _bool("WVC_RAPIDE_ALLER_RETOUR", False)
    mt_src: str = os.getenv("WVC_MT_SRC", "wol_Latn")
    mt_tgt: str = os.getenv("WVC_MT_TGT", "fra_Latn")
    tts_model: str = os.getenv("WVC_TTS_MODEL", "facebook/mms-tts-fra")
    device: str = os.getenv("WVC_DEVICE", "auto")  # auto | cpu | cuda
    mock_models: bool = _bool("WVC_MOCK_MODELS", False)  # tests / CI sans GPU
    # CPU : quantification int8 des couches linéaires de l'ASR (≈ 2× plus rapide).
    # 0 = jamais, 1 = modèles CTC (défaut), 2 = aussi Whisper.
    asr_int8_level: int = int(os.getenv("WVC_ASR_INT8", "1"))

    # --- Gemma via Ollama (WVC_MT_BACKEND=gemma) -----------------------------
    ollama_url: str = _OLLAMA_URL
    gemma_model: str = _GEMMA_MODEL               # ex. gemma4:e2b, gemma4:e4b, gemma3:4b
    gemma_temperature: float = float(os.getenv("WVC_GEMMA_TEMPERATURE", "0.7"))  # candidats 2..n
    # Rétro-traduction fr → wo : double le temps de traduction ; désactivée par défaut.
    gemma_roundtrip: bool = _bool("WVC_GEMMA_ROUNDTRIP", False)
    # Exemples dans l'invite : quelques-uns fixes (début d'invite identique à
    # chaque appel → mis en cache par Ollama, donc gratuits) + quelques-uns
    # proches de la phrase (recalculés à chaque appel).
    gemma_fixed_examples: int = int(os.getenv("WVC_GEMMA_EXEMPLES_FIXES", "3"))
    gemma_examples: int = int(os.getenv("WVC_GEMMA_EXEMPLES", "2"))
    # Les segments d'un même enregistrement sont traduits en un seul appel à Gemma.
    gemma_merge_segments: bool = _bool("WVC_GEMMA_FUSION", True)
    gemma_timeout_s: float = float(os.getenv("WVC_GEMMA_TIMEOUT_S", "180"))
    gemma_auto_pull: bool = _bool("WVC_GEMMA_AUTO_PULL", True)   # télécharge le modèle s'il manque

    # --- Réduction d'erreurs -------------------------------------------------
    num_beams: int = int(os.getenv("WVC_NUM_BEAMS", "5"))
    n_best: int = int(os.getenv("WVC_N_BEST", "3"))  # candidats reclassés
    rerank_alpha: float = float(os.getenv("WVC_RERANK_ALPHA", "0.5"))
    punctuation_restore: bool = _bool("WVC_PUNCT_RESTORE", True)
    vad_max_chunk_s: float = float(os.getenv("WVC_VAD_MAX_CHUNK_S", "20"))
    vad_min_silence_s: float = float(os.getenv("WVC_VAD_MIN_SILENCE_S", "0.35"))

    # --- Robustesse au bruit -------------------------------------------------
    # Force de la réduction de bruit (0 = désactivée, 1.5 = défaut, 2.5 = forte).
    # Réduction douce par défaut : trop forte, elle déforme la voix et la
    # reconnaissance se trompe davantage (le navigateur filtre déjà le bruit).
    denoise_strength: float = float(os.getenv("WVC_DEBRUITAGE", "1.0"))
    denoise_floor: float = float(os.getenv("WVC_DEBRUITAGE_PLANCHER", "0.3"))
    # Rapport signal/bruit minimal (dB) d'une zone pour être considérée comme parole.
    # Durée minimale d'une zone de parole : les bruits plus courts (clic, toux,
    # porte) sont ignorés.
    vad_min_speech_s: float = float(os.getenv("WVC_VAD_MIN_PAROLE_S", "0.2"))  # « waaw » ≈ 0,3 s
    # Marge gardée avant/après chaque zone de parole : évite de couper le début
    # ou la fin des mots (consonnes finales, « -ma », « -nga »…).
    vad_pad_s: float = float(os.getenv("WVC_VAD_MARGE_S", "0.3"))
    min_snr_db: float = float(os.getenv("WVC_SNR_MIN_DB", "10"))
    # Confiance ASR minimale d'un segment ; en dessous, c'est du bruit : ignoré.
    asr_min_conf: float = float(os.getenv("WVC_ASR_CONF_MIN", "0.45"))

    # --- Seuils de qualité (score estimé 0..1) -------------------------------
    quality_good: float = float(os.getenv("WVC_QUALITY_GOOD", "0.70"))
    quality_poor: float = float(os.getenv("WVC_QUALITY_POOR", "0.45"))
    # Seuil de cohérence aller-retour (chrF 0..100) sous lequel la traduction
    # est signalée « incertaine ». La rétro-traduction fr→wo (NLLB comme Gemma)
    # est faible : un seuil trop haut signale presque tout.
    roundtrip_min: float = float(os.getenv("WVC_ROUNDTRIP_MIN", "12"))

    # --- Limites et sécurité -------------------------------------------------
    max_audio_s: float = float(os.getenv("WVC_MAX_AUDIO_S", "120"))
    max_upload_mb: float = float(os.getenv("WVC_MAX_UPLOAD_MB", "10"))
    api_keys: list[str] = field(default_factory=lambda: _list("WVC_API_KEYS"))
    cors_origins: list[str] = field(default_factory=lambda: _list("WVC_CORS_ORIGINS"))
    rate_limit_per_min: int = int(os.getenv("WVC_RATE_LIMIT_PER_MIN", "30"))
    max_concurrent: int = int(os.getenv("WVC_MAX_CONCURRENT", "1"))
    max_queue: int = int(os.getenv("WVC_MAX_QUEUE", "4"))

    # --- Stockage / journalisation ------------------------------------------
    data_dir: Path = Path(os.getenv("WVC_DATA_DIR", "./runtime"))
    keep_audio: bool = _bool("WVC_KEEP_AUDIO", True)

    # --- Assistant conversationnel optionnel (Gemma via Ollama) -------------
    llm_enabled: bool = _bool("WVC_LLM_ENABLED", False)
    llm_url: str = os.getenv("WVC_LLM_URL", f"{_OLLAMA_URL}/api/generate")
    llm_model: str = os.getenv("WVC_LLM_MODEL", _GEMMA_MODEL)

    @property
    def mt_effectif(self) -> str:
        return {"gemma": self.gemma_model, "rapide": self.rapide_model}.get(self.mt_backend, self.mt_model)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "inferences.sqlite3"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"


settings = Settings()
