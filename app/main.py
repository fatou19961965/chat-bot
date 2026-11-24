"""API du chatbot vocal wolof → français.

Durcissement demandé par la phase 4 du sujet :
  * clé d'API (en-tête X-API-Key) si WVC_API_KEYS est défini ;
  * limitation de débit par clé / IP ;
  * file d'attente bornée et renvoi 429 sous charge (plus d'empilement silencieux) ;
  * limites effectives de taille et de durée ;
  * CORS restreint aux origines déclarées.
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import pipeline
from .config import settings
from .models import get_asr, get_translator, get_tts, loaded
from .memory import get_memory
from .quality import measured
from .storage import Store

logging.basicConfig(level=os.getenv("WVC_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("wvc.api")
STATIC = Path(__file__).parent / "static"


# --------------------------------------------------------------------------
# Contrôle de charge
# --------------------------------------------------------------------------
class Gate:
    """Sémaphore + file d'attente bornée : au-delà, 429 immédiat."""

    def __init__(self, max_concurrent: int, max_queue: int):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.waiting = 0
        self.max_queue = max_queue

    @asynccontextmanager
    async def slot(self):
        if self.sem.locked() and self.waiting >= self.max_queue:
            raise HTTPException(429, "Serveur occupé : réessayez dans quelques secondes.",
                                headers={"Retry-After": "5"})
        self.waiting += 1
        try:
            await self.sem.acquire()
        finally:
            self.waiting -= 1
        try:
            yield
        finally:
            self.sem.release()


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, "Trop de requêtes : limite par minute atteinte.",
                                headers={"Retry-After": "30"})
        q.append(now)


gate = Gate(settings.max_concurrent, settings.max_queue)
limiter = RateLimiter(settings.rate_limit_per_min)
store: Store | None = None


def get_store() -> Store:
    global store
    if store is None:
        store = Store()
    return store


def auth(request: Request) -> str:
    key = request.headers.get("x-api-key", "")
    if settings.api_keys and key not in settings.api_keys:
        raise HTTPException(401, "Clé d'API absente ou invalide (en-tête X-API-Key).")
    ident = key or (request.client.host if request.client else "anonyme")
    limiter.check(ident)
    return ident


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()
    if os.getenv("WVC_PRELOAD", "0") == "1":
        log.info("Préchargement des modèles…")
        await run_in_threadpool(get_asr)
        await run_in_threadpool(get_translator)
        await run_in_threadpool(get_tts)
    yield


app = FastAPI(title="Chatbot vocal wolof → français", version="1.0.0", lifespan=lifespan)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
@app.get("/static/img/hero.jpg", include_in_schema=False)
async def hero_photo():
    """Photo de bandeau facultative : sans photo, réponse vide (pas de « 404 Not Found »)."""
    f = STATIC / "img" / "hero.jpg"
    return FileResponse(f) if f.exists() else Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {
        "statut": "ok",
        "modeles": {
            "asr": settings.asr_model, "mt": settings.mt_effectif,
            "moteur_traduction": settings.mt_backend, "tts": settings.tts_model,
            "simulation": settings.mock_models,
        },
        "charges": loaded(),
        "prechargement": os.getenv("WVC_PRELOAD", "0") == "1",
        "auth": bool(settings.api_keys),
        "file_attente": gate.waiting,
    }


@app.post("/api/pipeline")
async def run_pipeline(
    audio: UploadFile = File(...),
    tts: bool = Form(True),
    ident: str = Depends(auth),
):
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    data = await audio.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Fichier trop volumineux (limite {settings.max_upload_mb:g} Mo).")
    async with gate.slot():
        try:
            return await run_in_threadpool(
                pipeline.run, data, get_store(), pipeline.Options(tts=tts), audio.filename or "audio.webm"
            )
        except pipeline.PipelineError as e:
            raise HTTPException(422, str(e)) from e


class TextIn(BaseModel):
    texte: str = Field(..., min_length=1, max_length=2000)


@app.post("/api/translate")
async def translate(body: TextIn, ident: str = Depends(auth)):
    async with gate.slot():
        return await run_in_threadpool(pipeline.translate_text, body.texte, get_store())


@app.post("/api/tts")
async def tts_endpoint(body: TextIn, ident: str = Depends(auth)):
    """Voix française du serveur (MMS-TTS) : utilisée quand le navigateur n'a pas
    de voix française ou n'arrive pas à parler (poste d'entreprise, proxy)."""
    from . import audio as A

    def _synth() -> bytes:
        out, sr = get_tts().synthesize(body.texte[:1000])
        return A.to_wav_bytes(out, sr)

    try:
        wav = await run_in_threadpool(_synth)
    except Exception as e:
        log.warning("TTS en échec : %s", e)
        raise HTTPException(503, "Synthèse vocale indisponible.") from e
    return Response(wav, media_type="audio/wav")


class CorrectionIn(BaseModel):
    ref_transcription: str | None = Field(None, max_length=4000)
    ref_traduction: str | None = Field(None, max_length=4000)


@app.post("/api/inferences/{iid}/correction")
def correct(iid: str, body: CorrectionIn, ident: str = Depends(auth)):
    s = get_store()
    rec = s.get(iid)
    if not rec:
        raise HTTPException(404, "Énoncé introuvable.")
    if not (body.ref_transcription or body.ref_traduction):
        raise HTTPException(422, "Fournissez au moins une transcription ou une traduction corrigée.")
    # Pour une saisie clavier, la « transcription » est le texte tapé : un WER
    # n'aurait pas de sens (ce n'est pas une erreur de reconnaissance).
    tape = isinstance(rec.get("models"), dict) and rec["models"].get("asr") == "saisie clavier"
    m = measured(rec["transcript"] or "", rec["translation"] or "",
                 None if tape else body.ref_transcription, body.ref_traduction)
    fields = {"wer": m.get("wer"), "cer": m.get("cer"), "chrfpp": m.get("chrfpp")}
    if body.ref_transcription:
        fields["ref_transcript"] = body.ref_transcription
    if body.ref_traduction:
        fields["ref_translation"] = body.ref_traduction
    s.correct(iid, {k: v for k, v in fields.items() if v is not None or k.startswith("ref_")})
    # La paire corrigée enrichit la mémoire de traduction (segments courts).
    appris = False
    if body.ref_traduction:
        wolof = body.ref_transcription or rec.get("transcript_post") or rec.get("transcript") or ""
        appris = get_memory().add(wolof, body.ref_traduction)
    return {"id": iid, "mesures": m, "memoire_enrichie": appris}


@app.get("/api/memoire")
def memoire(ident: str = Depends(auth)):
    tm = get_memory()
    return {"entrees": len(tm), "fichier_utilisateur": str(tm.user)}


@app.get("/api/inferences")
def list_inferences(limit: int = 50, ident: str = Depends(auth)):
    return get_store().recent(min(max(limit, 1), 500))


@app.get("/api/stats")
def stats(ident: str = Depends(auth)):
    return get_store().stats()


@app.get("/api/export.csv")
def export_csv(ident: str = Depends(auth)):
    """Paires corrigées au format manifeste du harnais (réutilisables en test
    ou en fine-tuning, après revue par un annotateur)."""
    rows = [r for r in get_store().recent(100_000) if r.get("ref_transcript") or r.get("ref_translation")]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["audio_path", "reference", "reference_fr", "hypothese", "traduction", "wer", "chrfpp", "source"])
    for r in rows:
        w.writerow([r.get("audio_path"), r.get("ref_transcript") or "", r.get("ref_translation") or "",
                    r.get("transcript"), r.get("translation"), r.get("wer"), r.get("chrfpp"), "usage"])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=corrections.csv"})
