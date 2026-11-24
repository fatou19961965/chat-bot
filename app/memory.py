"""Mémoire de traduction wolof → français.

Avant d'appeler NLLB, chaque segment est cherché dans la mémoire :
  * correspondance EXACTE après normalisation (casse, ponctuation, diacritiques
    ignorés) → traduction de la mémoire, confiance 1.0 ;
  * correspondance APPROCHÉE (similarité ≥ seuil, pour les fautes de frappe ou
    de reconnaissance : « naka nua def » → « naka nga def ») → confiance
    égale à la similarité.

Deux sources :
  * data/memoire_traduction.tsv   : liste validée, versionnée avec le projet ;
  * <WVC_DATA_DIR>/memoire_utilisateur.tsv : paires ajoutées automatiquement
    quand un utilisateur enregistre une correction. À relire périodiquement
    avant de les promouvoir dans la liste validée.

NLLB rend mal les expressions figées (salutations, formules) : c'est là que
la mémoire apporte le plus.
"""
from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .config import settings
from .metrics import normalize_text

CURATED = Path(__file__).resolve().parent.parent / "data" / "memoire_traduction.tsv"
FUZZY_MIN = 0.85       # similarité minimale pour une correspondance approchée
FUZZY_MIN_LEN = 6      # pas d'approché sur des clés trop courtes (faux positifs)


def key(text: str) -> str:
    return normalize_text(text, strip_diacritics=True).replace("-", " ").strip()


@dataclass
class Match:
    wolof: str
    francais: str
    similarite: float   # 1.0 = exacte
    origine: str        # "validée" | "utilisateur"

    @property
    def exacte(self) -> bool:
        return self.similarite >= 0.999


class TranslationMemory:
    def __init__(self, curated: Path = CURATED, user: Path | None = None):
        self.curated = Path(curated)
        self.user = Path(user or settings.data_dir / "memoire_utilisateur.tsv")
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, str, str]] = {}
        self.reload()

    def reload(self) -> None:
        entries: dict[str, tuple[str, str, str]] = {}
        # l'utilisateur d'abord, puis la liste validée qui a priorité
        for path, origin in ((self.user, "utilisateur"), (self.curated, "validée")):
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="\t"):
                    if len(row) < 2 or not row[0].strip() or row[0].startswith("#"):
                        continue
                    k = key(row[0])
                    if k:
                        entries[k] = (row[0].strip(), row[1].strip(), origin)
        with self._lock:
            self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, text: str) -> Match | None:
        k = key(text)
        if not k:
            return None
        with self._lock:
            hit = self._entries.get(k)
            if hit:
                return Match(hit[0], hit[1], 1.0, hit[2])
            if len(k) < FUZZY_MIN_LEN:
                return None
            best, best_r = None, 0.0
            for ek, val in self._entries.items():
                if len(ek) < FUZZY_MIN_LEN or abs(len(ek) - len(k)) > max(3, len(k) // 4):
                    continue
                r = SequenceMatcher(None, k, ek).ratio()
                if r > best_r:
                    best, best_r = val, r
        if best and best_r >= FUZZY_MIN:
            return Match(best[0], best[1], round(best_r, 3), best[2])
        return None

    def examples(self, text: str, k: int = 8) -> list[tuple[str, str]]:
        """Les k paires (wolof, français) les plus proches de `text` : exemples
        donnés à Gemma dans l'invite (apprentissage en contexte)."""
        if k <= 0:
            return []
        q = key(text)
        with self._lock:
            vals = list(self._entries.items())
        if not vals:
            return []
        scored = sorted(
            vals,
            key=lambda kv: (SequenceMatcher(None, q, kv[0]).ratio() if q else 0.0, kv[1][2] == "validée"),
            reverse=True,
        )
        return [(v[0], v[1]) for _, v in scored[:k]]

    def fixed_examples(self, k: int = 3) -> list[tuple[str, str]]:
        """k paires toujours identiques (les premières de la liste de départ) :
        elles forment un début d'invite stable qu'Ollama garde en cache."""
        with self._lock:
            vals = list(self._entries.values())
        base = [v for v in vals if v[2] != "utilisateur"] or vals
        return [(v[0], v[1]) for v in base[:k]]

    def add(self, wolof: str, francais: str) -> bool:
        """Ajoute (ou met à jour) une paire issue d'une correction utilisateur."""
        wolof, francais = wolof.strip(), francais.strip()
        k = key(wolof)
        if not k or not francais or len(k.split()) > 12:
            return False  # on ne mémorise que des segments courts
        with self._lock:
            current = self._entries.get(k)
            if current and current[2] == "validée":
                return False  # la liste validée ne se modifie pas automatiquement
            if current and current[1] == francais:
                return False
            self.user.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.user.exists()
            with self.user.open("a", encoding="utf-8", newline="") as f:
                if new_file:
                    f.write("# wolof\tfrançais — ajouts issus des corrections, à relire\n")
                csv.writer(f, delimiter="\t").writerow([wolof, francais])
            self._entries[k] = (wolof, francais, "utilisateur")
        return True


_tm: TranslationMemory | None = None


def get_memory() -> TranslationMemory:
    global _tm
    if _tm is None:
        _tm = TranslationMemory()
    return _tm
