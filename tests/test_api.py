"""Tests de bout en bout de l'API avec modèles simulés (WVC_MOCK_MODELS=1)."""
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app import main
    from app.storage import Store

    main.store = Store(tmp_path / "db.sqlite3")
    return TestClient(main.app)


def _wav():
    from app.audio import SR, to_wav_bytes

    t = np.arange(int(1.5 * SR)) / SR
    x = np.concatenate([np.zeros(SR // 2), 0.4 * np.sin(2 * np.pi * 220 * t), np.zeros(SR // 2)])
    return to_wav_bytes(x.astype(np.float32), SR)


def test_health(client):
    assert client.get("/api/health").json()["statut"] == "ok"


def test_pipeline_et_correction(client):
    r = client.post("/api/pipeline", files={"audio": ("a.wav", _wav(), "audio/wav")}, data={"tts": "true"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["traduction"] == "Bonjour, comment vas-tu ?"
    assert j["qualite"]["level"] in {"bon", "moyen", "faible"}
    assert j["audio_fr_wav_b64"]
    c = client.post(f"/api/inferences/{j['id']}/correction",
                    json={"ref_transcription": "Salaamaalekum, naka nga def?", "ref_traduction": "Bonjour, comment vas-tu ?"})
    assert c.status_code == 200 and c.json()["mesures"]["wer"] == 0.0
    assert client.get("/api/stats").json()["mesure"]["enonces_corriges"] == 1


def test_audio_invalide(client):
    r = client.post("/api/pipeline", files={"audio": ("a.webm", b"pas de l'audio", "audio/webm")})
    assert r.status_code == 422
