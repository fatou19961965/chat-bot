"""Journalisation des inférences (SQLite, bibliothèque standard).

Sur une langue peu dotée, c'est le seul mécanisme qui fait croître l'actif :
chaque énoncé stocke l'audio, la transcription, la traduction, les signaux de
confiance, et — si l'utilisateur la saisit — la correction, qui devient une
paire de référence pour l'évaluation et le fine-tuning.

Données personnelles : l'audio est une donnée biométrique potentielle.
Consentement, durée de conservation et déclaration CDP : voir docs/licences_et_conformite.md.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS inferences (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    audio_path TEXT,
    duration_s REAL,
    n_segments INTEGER,
    transcript TEXT,
    transcript_post TEXT,
    translation TEXT,
    candidates TEXT,
    asr_conf REAL,
    mt_conf REAL,
    roundtrip REAL,
    code_switch REAL,
    quality REAL,
    level TEXT,
    timings TEXT,
    models TEXT,
    ref_transcript TEXT,
    ref_translation TEXT,
    wer REAL,
    cer REAL,
    chrfpp REAL,
    corrected_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_ts ON inferences(ts);
"""


class Store:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_audio(self, inference_id: str, data: bytes, ext: str = "webm") -> str | None:
        if not settings.keep_audio:
            return None
        settings.audio_dir.mkdir(parents=True, exist_ok=True)
        p = settings.audio_dir / f"{inference_id}.{ext}"
        p.write_bytes(data)
        return str(p)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    def insert(self, rec: dict) -> None:
        rec = dict(rec)
        rec.setdefault("ts", time.time())
        for k in ("candidates", "timings", "models"):
            if k in rec and not isinstance(rec[k], str):
                rec[k] = json.dumps(rec[k], ensure_ascii=False)
        cols = ", ".join(rec)
        qs = ", ".join("?" for _ in rec)
        with self._conn() as c:
            c.execute(f"INSERT INTO inferences ({cols}) VALUES ({qs})", list(rec.values()))

    def get(self, inference_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM inferences WHERE id = ?", (inference_id,)).fetchone()
        return _row(row) if row else None

    def correct(self, inference_id: str, fields: dict) -> None:
        fields = {**fields, "corrected_ts": time.time()}
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE inferences SET {sets} WHERE id = ?", [*fields.values(), inference_id])

    def recent(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM inferences ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            g = c.execute(
                """SELECT COUNT(*) n, AVG(quality) q, AVG(asr_conf) a, AVG(mt_conf) m,
                          AVG(roundtrip) r, SUM(duration_s) d,
                          SUM(level='bon') bon, SUM(level='moyen') moyen, SUM(level='faible') faible
                   FROM inferences"""
            ).fetchone()
            m = c.execute(
                """SELECT COUNT(*) n, AVG(wer) wer, AVG(cer) cer, AVG(chrfpp) chrfpp
                   FROM inferences WHERE ref_transcript IS NOT NULL OR ref_translation IS NOT NULL"""
            ).fetchone()
            # corrélation grossière score estimé ↔ WER mesuré (étalonnage)
            pairs = c.execute(
                "SELECT quality, wer FROM inferences WHERE wer IS NOT NULL AND quality IS NOT NULL"
            ).fetchall()
        return {
            "estime": {
                "enonces": g["n"] or 0,
                "duree_audio_s": round(g["d"] or 0, 1),
                "score_moyen": _r(g["q"]),
                "confiance_asr_moyenne": _r(g["a"]),
                "confiance_mt_moyenne": _r(g["m"]),
                "aller_retour_moyen": _r(g["r"], 2),
                "repartition": {"bon": g["bon"] or 0, "moyen": g["moyen"] or 0, "faible": g["faible"] or 0},
            },
            "mesure": {
                "enonces_corriges": m["n"] or 0,
                "wer_moyen": _r(m["wer"]),
                "cer_moyen": _r(m["cer"]),
                "chrfpp_moyen": _r(m["chrfpp"], 2),
                "correlation_score_wer": _pearson([(p[0], p[1]) for p in pairs]),
            },
        }


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _pearson(pairs: list[tuple[float, float]]):
    if len(pairs) < 5:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if vx == 0 or vy == 0 else round(cov / (vx * vy), 3)


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("candidates", "timings", "models"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d
