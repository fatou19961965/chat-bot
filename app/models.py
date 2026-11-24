"""Modèles : reconnaissance (wolof), traduction (wolof → français), synthèse (français).

Traduction : deux moteurs interchangeables (WVC_MT_BACKEND)
  * gemma (défaut) : LLM Gemma servi localement par Ollama, guidé par des
    exemples tirés de la mémoire de traduction ;
  * nllb           : NLLB-200 via transformers (moteur d'origine).

Chaque étage renvoie, en plus de sa sortie, un signal de confiance interne :
  * ASR (CTC)     : probabilité moyenne du symbole retenu sur les trames non-blanc ;
  * ASR (Whisper) : probabilité moyenne par jeton généré ;
  * Traduction    : exp(log-prob moyenne par jeton) du candidat retenu ;
  * Aller-retour  : chrF entre la transcription wolof et la rétro-traduction
                    français → wolof du candidat (cohérence sémantique).

Les imports lourds (torch, transformers) sont différés : les tests et la CI
tournent avec WVC_MOCK_MODELS=1 sans les installer.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .metrics import chrf

log = logging.getLogger("wvc.models")
_lock = threading.Lock()      # état src_lang du tokenizer NLLB
_reg_lock = threading.Lock()  # chargement unique des modèles


def _device() -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ==========================================================================
# ASR
# ==========================================================================
@dataclass
class ASRResult:
    text: str
    confidence: float  # 0..1


class ASR:
    @property
    def min_conf(self) -> float:
        """Seuil « bruit » adapté au modèle : la confiance de Whisper (probabilité
        moyenne des jetons) est naturellement plus basse que celle d'un CTC."""
        base = settings.asr_min_conf
        return base * 0.5 if self.kind == "whisper" else base

    def __init__(self, model_id: str, lang: str = "wol"):
        self.model_id = model_id
        self.lang = lang
        self.kind = self._detect_kind(model_id)
        self.device = _device()
        self._load()

    @staticmethod
    def _peft_base(model_id: str) -> str | None:
        """Si `model_id` est un adaptateur PEFT/LoRA (ex. CAYTU/whosper-large-v2),
        renvoie l'identifiant du modèle de base sur lequel il se branche."""
        try:
            from huggingface_hub import hf_hub_download
            import json
            import os

            path = (os.path.join(model_id, "adapter_config.json") if os.path.isdir(model_id)
                    else hf_hub_download(model_id, "adapter_config.json"))
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("base_model_name_or_path")
        except Exception as e:
            if "whosper" in model_id.lower():
                log.warning("adapter_config.json illisible (%s) : base supposée openai/whisper-large-v2", e)
                return "openai/whisper-large-v2"
            return None

    @staticmethod
    def _detect_kind(model_id: str) -> str:
        """Whisper (encodeur-décodeur) ou CTC (wav2vec2, MMS, XLS-R, w2v-BERT),
        d'après la configuration du modèle et non son nom (ex. « whosper »)."""
        if "whisper" in model_id.lower() or "whosper" in model_id.lower():
            return "whisper"
        try:
            from transformers import AutoConfig

            return "whisper" if AutoConfig.from_pretrained(model_id).model_type == "whisper" else "ctc"
        except Exception:
            return "ctc"

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        log.info("Chargement ASR %s (%s) sur %s", self.model_id, self.kind, self.device)
        if self.kind == "ctc":
            from transformers import AutoModelForCTC

            import os

            kw = {}
            local_adapter = os.path.exists(os.path.join(self.model_id, f"adapter.{self.lang}.safetensors"))
            if "mms-1b" in self.model_id or local_adapter:  # modèle à adaptateurs
                kw = {"target_lang": self.lang, "ignore_mismatched_sizes": True}
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, **({"target_lang": self.lang} if kw else {})
            )
            self.model = AutoModelForCTC.from_pretrained(self.model_id, **kw)
            if kw:
                self.processor.tokenizer.set_target_lang(self.lang)
                self.model.load_adapter(self.lang)
            tok = getattr(self.processor, "tokenizer", self.processor)
            self.blank_id = tok.pad_token_id
        else:
            from transformers import WhisperForConditionalGeneration

            base = self._peft_base(self.model_id)
            if base:
                # Adaptateur PEFT : modèle de base + poids de l'adaptateur, fusionnés.
                try:
                    from peft import PeftModel
                except ImportError as e:
                    raise RuntimeError(
                        f"{self.model_id} est un adaptateur PEFT : installez-le avec "
                        "« pip install peft »."
                    ) from e
                log.info("Adaptateur PEFT %s sur %s", self.model_id, base)
                try:
                    self.processor = AutoProcessor.from_pretrained(self.model_id)
                except Exception:
                    self.processor = AutoProcessor.from_pretrained(base)
                model = WhisperForConditionalGeneration.from_pretrained(base)
                self.model = PeftModel.from_pretrained(model, self.model_id).merge_and_unload()
            else:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device).eval()
        if self.device == "cpu":
            import os

            torch.set_num_threads(max(1, os.cpu_count() or 1))
            # int8 : sûr pour les modèles CTC ; pour Whisper, le décodeur peut se
            # dégrader (mots inventés), donc seulement sur demande (WVC_ASR_INT8=2).
            if settings.asr_int8_level >= (2 if self.kind == "whisper" else 1):
                try:
                    self.model = torch.ao.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8)
                    log.info("ASR quantifié en int8 (CPU).")
                except Exception as e:  # dépend de la version de torch
                    log.warning("Quantification int8 impossible (%s) : modèle en float32.", e)

    def transcribe(self, audio: np.ndarray, sr: int = 16_000) -> ASRResult:
        import torch

        if audio.size < sr * 0.1:
            return ASRResult("", 0.0)
        if self.kind == "ctc":
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits.float(), dim=-1)
            conf_frames, ids = probs.max(dim=-1)
            text = self.processor.decode(ids.cpu())
            mask = ids != self.blank_id
            conf = float(conf_frames[mask].mean()) if bool(mask.any()) else 0.0
            return ASRResult(text.strip(), conf)

        feats = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(
            self.device, dtype=self.model.dtype)
        # Budget de jetons proportionnel à la durée : Whisper s'arrête plus tôt
        # et ne peut pas partir en boucle sur un long texte inventé.
        budget = int(min(220, 8 * len(audio) / sr + 16))
        with torch.inference_mode():
            out = self.model.generate(
                feats, return_dict_in_generate=True, output_scores=True, max_new_tokens=budget,
                num_beams=1,
            )
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        text = " ".join(text.split())
        trans = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        lp = trans[0][torch.isfinite(trans[0])]
        conf = float(torch.exp(lp.mean())) if lp.numel() else 0.0
        return ASRResult(text.strip(), conf)


class MockASR:
    model_id = "mock-asr"
    min_conf = 0.45

    def transcribe(self, audio: np.ndarray, sr: int = 16_000) -> ASRResult:
        if audio.size < sr * 0.1:
            return ASRResult("", 0.0)
        return ASRResult("salaamaalekum naka nga def", 0.86)


# ==========================================================================
# Traduction
# ==========================================================================
@dataclass
class Candidate:
    text: str
    logprob: float            # log-prob moyenne par jeton
    roundtrip: float = 0.0    # chrF aller-retour 0..100
    score: float = 0.0        # score de reclassement 0..1

    @property
    def confidence(self) -> float:
        return math.exp(self.logprob)


@dataclass
class MTResult:
    best: Candidate
    candidates: list[Candidate] = field(default_factory=list)


class Translator:
    backend = "nllb"

    def __init__(self, model_id: str, src: str, tgt: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_id, self.src, self.tgt = model_id, src, tgt
        self.device = _device()
        log.info("Chargement traduction %s sur %s", model_id, self.device)
        self.tok = AutoTokenizer.from_pretrained(model_id, src_lang=src)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device).eval()

    def _generate(self, texts: list[str], src: str, tgt: str, beams: int, n_ret: int):
        import torch

        with _lock:  # le tokenizer NLLB porte src_lang en état
            self.tok.src_lang = src
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = enc.to(self.device)
        # Sortie bornée à ~2,5× la longueur d'entrée : coupe les traductions qui
        # « continuent » au-delà du sens (hallucinations par dérive).
        max_new = min(256, int(enc["input_ids"].shape[1] * 2.5) + 10)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt),
                num_beams=max(beams, n_ret),
                num_return_sequences=n_ret,
                max_length=max_new + 1,  # côté décodeur ; évite le conflit avec generation_config
                length_penalty=1.0,
                no_repeat_ngram_size=4,  # limite les boucles d'hallucination
                output_scores=True,
                return_dict_in_generate=True,
            )
        dec = self.tok.batch_decode(out.sequences, skip_special_tokens=True)
        scores = out.sequences_scores.tolist() if out.sequences_scores is not None else [0.0] * len(dec)
        return dec, scores

    def translate(self, text: str, n_best: int = 3, num_beams: int = 5, alpha: float = 0.5) -> MTResult:
        """Génère n candidats, les rétro-traduit, et retient celui qui maximise
        alpha * confiance_modèle + (1 - alpha) * cohérence_aller_retour."""
        if not text.strip():
            return MTResult(Candidate("", -10.0))
        dec, scores = self._generate([text], self.src, self.tgt, num_beams, n_best)
        cands = [Candidate(t.strip(), s) for t, s in zip(dec, scores)]
        # dédoublonnage (le beam renvoie parfois des variantes identiques)
        uniq: dict[str, Candidate] = {}
        for c in cands:
            if c.text and c.text not in uniq:
                uniq[c.text] = c
        cands = list(uniq.values()) or cands[:1]
        back, _ = self._generate([c.text for c in cands], self.tgt, self.src, 2, 1)
        for c, b in zip(cands, back):
            c.roundtrip = chrf(b, text)
            c.score = alpha * c.confidence + (1 - alpha) * c.roundtrip / 100.0
            # Rapport de longueur anormal (mots fr / mots wo) : signe classique
            # d'hallucination ou d'omission → candidat pénalisé.
            ratio = len(c.text.split()) / max(1, len(text.split()))
            if not 0.4 <= ratio <= 3.0:
                c.score *= 0.5
        cands.sort(key=lambda c: c.score, reverse=True)
        return MTResult(cands[0], cands)

    def translate_plain(self, texts: list[str], num_beams: int = 5, reverse: bool = False) -> list[str]:
        """Traduction simple par lot (benchmarks FLORES)."""
        src, tgt = (self.tgt, self.src) if reverse else (self.src, self.tgt)
        dec, _ = self._generate(texts, src, tgt, num_beams, 1)
        return [d.strip() for d in dec]


class MockTranslator:
    model_id = "mock-mt"
    backend = "simulation"
    _table = {"salaamaalekum naka nga def": "Bonjour, comment vas-tu ?"}

    def translate(self, text: str, n_best: int = 3, num_beams: int = 5, alpha: float = 0.5) -> MTResult:
        if not text.strip():
            return MTResult(Candidate("", -10.0))
        from .metrics import normalize_text

        fr = self._table.get(normalize_text(text), f"[fr] {text}")
        c = Candidate(fr, math.log(0.78), roundtrip=72.0)
        c.score = alpha * c.confidence + (1 - alpha) * 0.72
        return MTResult(c, [c])

    def translate_plain(self, texts, num_beams: int = 5, reverse: bool = False):
        return [self.translate(t).best.text for t in texts]


# --------------------------------------------------------------------------
# Traduction rapide : NLLB compressé en int8 (CTranslate2 si présent)
# --------------------------------------------------------------------------
class FastTranslator:
    """NLLB-200 600M optimisé pour le CPU : 10 à 30× plus rapide que Gemma.

    * CTranslate2 (si installé) : le modèle est converti une seule fois en int8
      dans `runtime/ct2/…`, puis traduit en quelques centaines de ms ;
    * sinon, transformers + quantification int8 dynamique (≈ 2× plus rapide
      que NLLB normal).
    Décodage glouton (1 faisceau), sortie bornée, cache des phrases déjà vues,
    pas de rétro-traduction par défaut (WVC_RAPIDE_ALLER_RETOUR=1 pour l'activer).
    """

    backend = "rapide"

    def __init__(self, model_id: str, src: str = "wol_Latn", tgt: str = "fra_Latn"):
        import os

        from transformers import AutoTokenizer

        self.model_id, self.src, self.tgt = model_id, src, tgt
        self.roundtrip_enabled = settings.rapide_roundtrip
        self._cache: dict[tuple, MTResult] = {}
        self.tok = AutoTokenizer.from_pretrained(model_id, src_lang=src)
        self.ct2 = None
        try:
            import ctranslate2

            out = settings.data_dir / "ct2" / (model_id.replace("/", "__") + "-int8")
            if not (out / "model.bin").exists():
                log.warning("Conversion de %s pour CTranslate2 (une seule fois, 1 à 3 min)…", model_id)
                out.parent.mkdir(parents=True, exist_ok=True)
                ctranslate2.converters.TransformersConverter(model_id).convert(
                    str(out), quantization="int8", force=True)
            self.ct2 = ctranslate2.Translator(str(out), device="cpu", compute_type="int8",
                                              intra_threads=max(1, os.cpu_count() or 1))
            log.info("Traduction rapide : %s via CTranslate2 int8", model_id)
        except Exception as e:  # ctranslate2 absent ou conversion impossible
            log.warning("CTranslate2 indisponible (%s) : NLLB int8 via transformers.", e)
            import torch
            from transformers import AutoModelForSeq2SeqLM

            torch.set_num_threads(max(1, os.cpu_count() or 1))
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()
            try:
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            except Exception as qe:
                log.warning("Quantification int8 impossible (%s).", qe)
            self.hf = model

    def _generate(self, texts: list[str], src: str, tgt: str, n: int = 1) -> tuple[list[list[str]], list[list[float]]]:
        with _lock:
            self.tok.src_lang = src
            enc = [self.tok.encode(t, truncation=True, max_length=256) for t in texts]
        max_len = [min(256, int(len(e) * 2.5) + 10) for e in enc]
        if self.ct2 is not None:
            # NB : translate_batch n'a pas d'option normalize_scores (contrairement à
            # Generator) : on divise nous-mêmes le score cumulé par la longueur.
            res = self.ct2.translate_batch(
                [self.tok.convert_ids_to_tokens(e) for e in enc],
                target_prefix=[[tgt]] * len(texts),
                beam_size=max(1, n), num_hypotheses=n, return_scores=True, length_penalty=0,
                max_decoding_length=max(max_len), no_repeat_ngram_size=4,
            )
            outs, scores = [], []
            for r in res:
                outs.append([self.tok.decode(self.tok.convert_tokens_to_ids(h[1:]), skip_special_tokens=True).strip()
                             for h in r.hypotheses])
                scores.append([float(sc) / max(1, len(h)) for sc, h in zip(r.scores, r.hypotheses)])
            return outs, scores
        import torch

        with _lock:
            self.tok.src_lang = src
            batch = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            out = self.hf.generate(
                **batch, forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt),
                num_beams=max(1, n), num_return_sequences=n, max_length=max(max_len) + 1,
                no_repeat_ngram_size=4, output_scores=True, return_dict_in_generate=True,
            )
        dec = self.tok.batch_decode(out.sequences, skip_special_tokens=True)
        if n == 1 and out.scores:
            tr = self.hf.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            sc = [float(row[torch.isfinite(row)].mean()) if torch.isfinite(row).any() else -1.0 for row in tr]
        else:
            sc = out.sequences_scores.tolist() if out.sequences_scores is not None else [-1.0] * len(dec)
        return ([[d.strip() for d in dec[i * n:(i + 1) * n]] for i in range(len(texts))],
                [sc[i * n:(i + 1) * n] for i in range(len(texts))])

    def translate(self, text: str, n_best: int = 1, num_beams: int = 1, alpha: float = 0.5) -> MTResult:
        if not text.strip():
            return MTResult(Candidate("", -10.0))
        ck = (text, n_best)
        if ck in self._cache:
            return self._cache[ck]
        outs, scores = self._generate([text], self.src, self.tgt, max(1, n_best))
        uniq: dict[str, Candidate] = {}
        for t, sc in zip(outs[0], scores[0]):
            if t and t not in uniq:
                uniq[t] = Candidate(t, float(sc))
        cands = list(uniq.values()) or [Candidate("", -10.0)]
        if self.roundtrip_enabled:
            back, _ = self._generate([c.text for c in cands], self.tgt, self.src, 1)
            for c, b in zip(cands, back):
                c.roundtrip = chrf(b[0], text)
        for c in cands:
            if not self.roundtrip_enabled:
                c.roundtrip = 100.0 * c.confidence
            c.score = alpha * c.confidence + (1 - alpha) * c.roundtrip / 100.0
            ratio = len(c.text.split()) / max(1, len(text.split()))
            if not 0.4 <= ratio <= 3.0:
                c.score *= 0.5
        cands.sort(key=lambda c: c.score, reverse=True)
        res = MTResult(cands[0], cands)
        if len(self._cache) > 1000:
            self._cache.clear()
        self._cache[ck] = res
        return res

    def translate_plain(self, texts: list[str], num_beams: int = 1, reverse: bool = False) -> list[str]:
        src, tgt = (self.tgt, self.src) if reverse else (self.src, self.tgt)
        outs, _ = self._generate(texts, src, tgt, 1)
        return [o[0] for o in outs]


# --------------------------------------------------------------------------
# Traduction par Gemma (Ollama)
# --------------------------------------------------------------------------
LANG_NAMES = {
    "wol_Latn": "wolof (orthographe CLAD, alphabet latin)",
    "fra_Latn": "français",
    "eng_Latn": "anglais",
}


class OllamaError(RuntimeError):
    pass


def clean_llm_output(text: str) -> str:
    """Ne garde que la traduction : retire réflexion, balises, étiquettes et guillemets."""
    import re

    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"```[a-zA-Z]*", "", t).replace("```", "")
    t = t.strip()
    # premier paragraphe seulement (le modèle ajoute parfois une explication)
    t = re.split(r"\n\s*\n", t, maxsplit=1)[0].strip()
    t = re.sub(r"^(\*\*)?(traduction|translation|français|french|wolof)\s*(\([^)]*\))?\s*:\s*(\*\*)?\s*",
               "", t, flags=re.I)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = " ".join(lines)
    pairs = (('"', '"'), ("«", "»"), ("“", "”"), ("'", "'"), ("*", "*"))
    for a, b in pairs:
        if len(t) > 2 and t.startswith(a) and t.endswith(b):
            t = t[len(a):-len(b)].strip()
    return t


class GemmaTranslator:
    """Traduction wolof → français par Gemma servi par Ollama (API HTTP locale).

    Mêmes sorties que `Translator` (n-best, confiance, aller-retour, score) :
      * candidat 1 en décodage glouton (température 0), candidats suivants
        échantillonnés (WVC_GEMMA_TEMPERATURE) ;
      * confiance = exp(log-prob moyenne par jeton) si Ollama renvoie les
        logprobs (Ollama ≥ 0.12.11), sinon accord entre candidats (chrF) ;
      * aller-retour = chrF entre le wolof d'origine et la rétro-traduction
        français → wolof par Gemma ;
      * l'invite contient les paires les plus proches de la mémoire de
        traduction (exemples validés par des locuteurs).
    """

    backend = "gemma"

    def __init__(self, model_id: str, src: str = "wol_Latn", tgt: str = "fra_Latn",
                 url: str | None = None, check: bool = True):
        self.model_id, self.src, self.tgt = model_id, src, tgt
        self.url = (url or settings.ollama_url).rstrip("/")
        self.timeout = settings.gemma_timeout_s
        self._think_ok = True      # le paramètre "think" est refusé par certaines versions
        self._logprobs_ok = True   # idem pour "logprobs"
        self.roundtrip_enabled = settings.gemma_roundtrip
        self._cache: dict[tuple, MTResult] = {}
        if check:
            self._ensure_model()
            self._warmup()

    def _warmup(self) -> None:
        """Charge Gemma en mémoire dès le démarrage (sinon la 1re traduction
        attend 20 à 60 s le chargement du modèle)."""
        try:
            self._http("/api/generate", {"model": self.model_id, "prompt": "ok", "stream": False,
                                         "keep_alive": "24h", "options": {"num_predict": 1}})
            log.info("Gemma chargé en mémoire.")
        except OllamaError as e:
            log.warning("Préchauffage de Gemma impossible : %s", e)

    # -- HTTP --------------------------------------------------------------
    def _http(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        import json
        import urllib.error
        import urllib.request

        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if payload is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise OllamaError(f"Ollama {e.code} : {body[:300]}") from e
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError(
                f"Ollama injoignable à {self.url} ({e}). Lancez Ollama (application ou "
                f"« ollama serve ») puis réessayez."
            ) from e

    def _ensure_model(self) -> None:
        tags = self._http("/api/tags", timeout=10)
        names = {m.get("name", "") for m in tags.get("models", [])}
        names |= {n.removesuffix(":latest") for n in names}
        if self.model_id in names:
            log.info("Traduction : %s via Ollama (%s)", self.model_id, self.url)
            return
        if not settings.gemma_auto_pull:
            raise OllamaError(f"Modèle {self.model_id} absent : lancez « ollama pull {self.model_id} ».")
        log.warning("Téléchargement de %s par Ollama (plusieurs Go, une seule fois)…", self.model_id)
        self._http("/api/pull", {"model": self.model_id, "stream": False}, timeout=6 * 3600)

    def _chat(self, messages: list[dict], temperature: float, seed: int,
              num_predict: int = 160) -> tuple[str, float | None]:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "keep_alive": "24h",
            "options": {"temperature": temperature, "seed": seed, "top_p": 0.95,
                        "num_predict": num_predict, "num_ctx": 1536},
        }
        if self._think_ok:
            payload["think"] = False
        if self._logprobs_ok:
            payload["logprobs"] = True
        try:
            out = self._http("/api/chat", payload)
        except OllamaError as e:
            msg = str(e).lower()
            if "400" in msg and ("think" in msg or "logprob" in msg):
                if "think" in msg:
                    self._think_ok = False
                else:
                    self._logprobs_ok = False
                return self._chat(messages, temperature, seed, num_predict)
            raise
        text = clean_llm_output((out.get("message") or {}).get("content", ""))
        lps = [x.get("logprob") for x in (out.get("logprobs") or []) if isinstance(x, dict)]
        lps = [v for v in lps if isinstance(v, (int, float)) and math.isfinite(v)]
        return text, (sum(lps) / len(lps) if lps else None)

    # -- Invite ------------------------------------------------------------
    def _messages(self, text: str, src: str, tgt: str) -> list[dict]:
        s_name, t_name = LANG_NAMES.get(src, src), LANG_NAMES.get(tgt, tgt)
        system = (
            f"Tu es un traducteur professionnel {s_name} → {t_name}, spécialiste du wolof parlé "
            "au Sénégal. Le texte source peut venir d'une reconnaissance vocale : il peut "
            "contenir des fautes d'orthographe et des mots français mêlés au wolof (à garder). "
            f"Traduis fidèlement, sans rien ajouter ni expliquer. Réponds UNIQUEMENT par la "
            f"traduction en {t_name}, sur une seule ligne."
        )
        msgs = [{"role": "system", "content": system}]
        from .memory import get_memory

        mem = get_memory()
        fixed = mem.fixed_examples(settings.gemma_fixed_examples) if settings.gemma_fixed_examples > 0 else []
        near = [p for p in (mem.examples(text, settings.gemma_examples + len(fixed))
                            if settings.gemma_examples > 0 else []) if p not in fixed]
        pairs = fixed + near[: settings.gemma_examples]   # fixes d'abord : préfixe stable
        if src == self.tgt:  # sens inverse (rétro-traduction)
            pairs = [(fr, wo) for wo, fr in pairs]
        for a, b in pairs:
            msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": text})
        return msgs

    # -- API commune avec Translator -----------------------------------------
    def translate(self, text: str, n_best: int = 3, num_beams: int = 5, alpha: float = 0.5) -> MTResult:
        if not text.strip():
            return MTResult(Candidate("", -10.0))
        ck = (text, n_best, alpha)
        if ck in self._cache:
            return self._cache[ck]
        msgs = self._messages(text, self.src, self.tgt)
        # Longueur de sortie bornée selon la phrase : le modèle s'arrête plus tôt.
        budget = min(320, 3 * len(text.split()) + 24)
        raw: list[tuple[str, float | None]] = []
        for i in range(max(1, n_best)):
            temp = 0.0 if i == 0 else settings.gemma_temperature
            raw.append(self._chat(msgs, temp, seed=i, num_predict=budget))
        uniq: dict[str, float | None] = {}
        for t, lp in raw:
            if t and t not in uniq:
                uniq[t] = lp
        if not uniq:
            return MTResult(Candidate("", -10.0))
        samples = [t for t, _ in raw if t]
        cands = []
        for t, lp in uniq.items():
            if lp is None:
                # Pas de logprobs : auto-cohérence = accord moyen (chrF) avec les
                # autres échantillons ; 0,6 (neutre) s'il n'y a qu'un échantillon.
                rest = list(samples)
                rest.remove(t)
                agree = sum(chrf(t, o) for o in rest) / (100.0 * len(rest)) if rest else 0.6
                lp = math.log(max(1e-3, min(1.0, agree)))
            cands.append(Candidate(t, lp))
        for c in cands:
            if self.roundtrip_enabled:
                back, _ = self._chat(self._messages(c.text, self.tgt, self.src), 0.0, seed=0,
                                     num_predict=budget)
                c.roundtrip = chrf(back, text)
                w_rt = 1 - alpha
            else:
                # Pas de rétro-traduction : la confiance du modèle sert d'indicateur.
                c.roundtrip = 100.0 * c.confidence
                w_rt = 0.0
            c.score = (alpha * c.confidence + w_rt * c.roundtrip / 100.0) / (alpha + w_rt or 1.0)
            ratio = len(c.text.split()) / max(1, len(text.split()))
            if not 0.4 <= ratio <= 3.0:
                c.score *= 0.5
        cands.sort(key=lambda c: c.score, reverse=True)
        res = MTResult(cands[0], cands)
        if len(self._cache) > 500:
            self._cache.clear()
        self._cache[ck] = res
        return res

    def translate_plain(self, texts: list[str], num_beams: int = 5, reverse: bool = False) -> list[str]:
        src, tgt = (self.tgt, self.src) if reverse else (self.src, self.tgt)
        return [self._chat(self._messages(t, src, tgt), 0.0, seed=0)[0] if t.strip() else "" for t in texts]


def is_ollama_model(model_id: str) -> bool:
    """« gemma4:e4b », « gemma3 », « ollama:xxx » → Ollama ; « facebook/nllb-… » → transformers."""
    m = model_id.lower()
    return m.startswith("ollama:") or ("/" not in m and (":" in m or m.startswith("gemma")))


def make_translator(model_id: str, src: str = "wol_Latn", tgt: str = "fra_Latn"):
    if is_ollama_model(model_id):
        return GemmaTranslator(model_id.removeprefix("ollama:"), src, tgt)
    return Translator(model_id, src, tgt)


# ==========================================================================
# Synthèse vocale (français)
# ==========================================================================
class TTS:
    def __init__(self, model_id: str):
        from transformers import AutoTokenizer, VitsModel

        self.device = _device()
        log.info("Chargement TTS %s sur %s", model_id, self.device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device).eval()
        self.sr = int(self.model.config.sampling_rate)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import re

        import torch

        parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
        pieces = []
        for p in parts:
            inputs = self.tok(p, return_tensors="pt").to(self.device)
            if inputs["input_ids"].shape[-1] == 0:
                continue
            torch.manual_seed(0)  # VITS est stochastique : sortie reproductible
            with torch.inference_mode():
                wav = self.model(**inputs).waveform[0].float().cpu().numpy()
            pieces += [wav, np.zeros(int(0.15 * self.sr), dtype=np.float32)]
        if not pieces:
            return np.zeros(1, dtype=np.float32), self.sr
        return np.concatenate(pieces), self.sr


class MockTTS:
    sr = 16_000

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        t = np.linspace(0, 0.4, int(0.4 * self.sr), endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), self.sr


# ==========================================================================
# Registre paresseux
# ==========================================================================
_asr = _mt = _tts = None


def get_asr():
    global _asr
    if _asr is None:
        with _reg_lock:
            if _asr is None:
                _asr = MockASR() if settings.mock_models else ASR(settings.asr_model, settings.asr_lang)
    return _asr


def get_translator():
    global _mt
    if _mt is None:
        with _reg_lock:
            if _mt is None:
                if settings.mock_models:
                    _mt = MockTranslator()
                elif settings.mt_backend == "gemma":
                    _mt = GemmaTranslator(settings.gemma_model, settings.mt_src, settings.mt_tgt)
                elif settings.mt_backend == "rapide":
                    _mt = FastTranslator(settings.rapide_model, settings.mt_src, settings.mt_tgt)
                else:
                    _mt = Translator(settings.mt_model, settings.mt_src, settings.mt_tgt)
    return _mt


def get_tts():
    global _tts
    if _tts is None:
        with _reg_lock:
            if _tts is None:
                _tts = MockTTS() if settings.mock_models else TTS(settings.tts_model)
    return _tts


def loaded() -> dict:
    """État de chargement, pour l'indicateur de l'interface."""
    return {"asr": _asr is not None, "mt": _mt is not None, "tts": _tts is not None}
