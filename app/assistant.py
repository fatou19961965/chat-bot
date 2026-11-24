"""Réponse conversationnelle optionnelle via Gemma servi localement par Ollama.

Désactivée par défaut (WVC_LLM_ENABLED=0) : le cœur du projet est la
traduction mesurée. Activée, elle reprend le comportement du prototype
(réponse de l'assistant en français, lue à voix haute côté navigateur).
"""
from __future__ import annotations

import json
import logging
import urllib.request

from .config import settings

log = logging.getLogger("wvc.assistant")

SYSTEM = (
    "Tu es un assistant bancaire courtois. L'utilisateur s'exprime en wolof ; "
    "son message t'arrive traduit automatiquement en français et peut contenir "
    "des erreurs. Si le sens est ambigu, demande une reformulation. Réponds en "
    "français, en deux phrases maximum."
)


def reply(user_fr: str, timeout: float = 30.0) -> str | None:
    payload = json.dumps({
        "model": settings.llm_model,
        "prompt": f"{SYSTEM}\n\nUtilisateur : {user_fr}\nAssistant :",
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode()
    req = urllib.request.Request(settings.llm_url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "").strip() or None
    except Exception as e:
        log.warning("LLM indisponible : %s", e)
        return None
