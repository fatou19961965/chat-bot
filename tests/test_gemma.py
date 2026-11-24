"""Traducteur Gemma : testé sans Ollama, avec un faux serveur HTTP en mémoire."""
import math

from app.models import GemmaTranslator, OllamaError, clean_llm_output, is_ollama_model


class FakeOllama:
    """Remplace GemmaTranslator._http : renvoie des réponses /api/chat scriptées."""

    def __init__(self, fr="Bonjour, comment vas-tu ?", back="salaamaalekum naka nga def",
                 logprobs=True, refuse_think=False):
        self.fr, self.back, self.logprobs, self.refuse_think = fr, back, logprobs, refuse_think
        self.calls = []

    def __call__(self, path, payload=None, timeout=None):
        self.calls.append((path, payload))
        if path == "/api/tags":
            return {"models": [{"name": "gemma4:e4b"}]}
        if self.refuse_think and payload.get("think") is not None:
            raise OllamaError('Ollama 400 : {"error":"\\"gemma4:e4b\\" does not support thinking"}')
        system = payload["messages"][0]["content"]
        to_fr = "→ français" in system
        out = {"message": {"role": "assistant",
                           "content": f"Traduction : « {self.fr} »" if to_fr else self.back}}
        if self.logprobs:
            out["logprobs"] = [{"token": "x", "logprob": -0.1}, {"token": "y", "logprob": -0.3}]
        return out


def _tr(fake):
    t = GemmaTranslator("gemma4:e4b", check=False)
    t._http = fake
    return t


def test_nettoyage_sortie():
    assert clean_llm_output("Traduction : « Bonjour. »") == "Bonjour."
    assert clean_llm_output("<think>hmm</think>\n**Français :** Merci.\n\nNote : formule polie.") == "Merci."
    assert clean_llm_output('"J\'ai faim."') == "J'ai faim."


def test_detection_modele_ollama():
    assert is_ollama_model("gemma4:e4b") and is_ollama_model("gemma3") and is_ollama_model("ollama:x")
    assert not is_ollama_model("facebook/nllb-200-distilled-600M")


def test_traduction_avec_logprobs_et_aller_retour():
    fake = FakeOllama()
    t = _tr(fake)
    t.roundtrip_enabled = True
    res = t.translate("salaamaalekum naka nga def", n_best=1)
    assert res.best.text == "Bonjour, comment vas-tu ?"
    assert math.isclose(res.best.confidence, math.exp(-0.2), rel_tol=1e-6)
    assert res.best.roundtrip > 90                     # rétro-traduction identique à la source
    assert 0 < res.best.score <= 1
    chat = [p for path, p in fake.calls if path == "/api/chat"]
    assert len(chat) == 2                               # traduction + rétro-traduction
    assert chat[0]["options"]["temperature"] == 0.0 and chat[0]["think"] is False


def test_exemples_de_la_memoire_dans_l_invite():
    fake = FakeOllama()
    _tr(fake).translate("naka nga def", n_best=1)
    msgs = fake.calls[0][1]["messages"]
    assert msgs[-1] == {"role": "user", "content": "naka nga def"}
    assert any(m["role"] == "assistant" for m in msgs[1:-1])   # paires tirées de data/memoire_traduction.tsv


def test_sans_logprobs_auto_coherence():
    fake = FakeOllama(logprobs=False)
    res = _tr(fake).translate("salaamaalekum naka nga def", n_best=3)
    # trois échantillons identiques → un seul candidat, accord parfait
    assert len(res.candidates) == 1 and res.best.confidence > 0.99


def test_version_ollama_sans_think():
    fake = FakeOllama(refuse_think=True)
    t = _tr(fake)
    assert t.translate("naka nga def", n_best=1).best.text == "Bonjour, comment vas-tu ?"
    assert t._think_ok is False


def test_traduction_par_lot_inverse():
    fake = FakeOllama()
    assert _tr(fake).translate_plain(["Bonjour"], reverse=True) == ["salaamaalekum naka nga def"]


def test_rapide_sans_aller_retour_et_cache():
    fake = FakeOllama()
    t = _tr(fake)
    t.roundtrip_enabled = False
    r1 = t.translate("naka nga def", n_best=1)
    r2 = t.translate("naka nga def", n_best=1)
    chat = [p for path, p in fake.calls if path == "/api/chat"]
    assert len(chat) == 1                    # un seul appel, puis le cache
    assert r1 is r2 and r1.best.roundtrip > 0
    assert chat[0]["options"]["num_predict"] < 100   # sortie bornée à la longueur de la phrase
