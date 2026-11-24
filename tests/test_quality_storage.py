from app.quality import estimate, measured
from app.storage import Store


def test_niveaux():
    assert estimate(0.95, 0.9, 90, n_words=6).level == "bon"
    low = estimate(0.2, 0.6, 60, n_words=6)
    assert low.level == "faible" and "reconnaissance" in low.advice


def test_alternance_penalise():
    assert estimate(0.8, 0.8, 80, 0.5, 6).score < estimate(0.8, 0.8, 80, 0.0, 6).score


def test_mesures_reelles():
    m = measured("naka nga def", "Comment vas-tu ?", "naka nga def", "Comment vas-tu ?")
    assert m["wer"] == 0.0 and m["chrfpp"] == 100.0


def test_store(tmp_path):
    s = Store(tmp_path / "db.sqlite3")
    s.insert({"id": "x1", "transcript": "a b", "translation": "c", "quality": 0.5, "level": "moyen",
              "asr_conf": 0.5, "mt_conf": 0.5, "roundtrip": 50, "candidates": [], "timings": {}, "models": {}})
    s.correct("x1", {"ref_transcript": "a c", "wer": 0.5})
    st = s.stats()
    assert st["estime"]["enonces"] == 1 and st["mesure"]["enonces_corriges"] == 1
    assert s.get("x1")["wer"] == 0.5
