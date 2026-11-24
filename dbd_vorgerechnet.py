"""Vorgerechnete Reihen: `tools/messreihen.json` (erzeugt von `tools/messreihen.py` mit dem Code der Demo, Zufalls-Seed 2026).

Die App rechnet die Reihen der Experimente 1 bis 3 und den Minibatch-Teil live. Vorgerechnet sind nur das, was die Seite zum Vergleichen und zum Belegen braucht: die Bezugsreihen
je Dimension (Experiment 1), die Messung bei n = 2 048 und 4 096 (Extrapolation), die Tabelle des Tabs „sliced“ und die Zahlen der README (Struktur mit 100 Wiederholungen,
Minibatch mit fünf Sätzen von Zufalls-Seeds). `tests/test_claims.py` rechnet Zellen davon neu und vergleicht als Band."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

DATEI = Path(__file__).resolve().parent / "tools" / "messreihen.json"


@lru_cache(maxsize=1)
def lade() -> dict:
    try:
        return json.loads(DATEI.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def teil(name: str):
    """Ein Teil der Datei oder None, wenn er fehlt."""
    return lade().get(name)


QUELLEN = ("dbd_sweeps.py", "dbd_zufall.py", "dbd_wasserstein.py", "dbd_sliced.py", "dbd_zuordnung.py", "dbd_rng.py")


def quellen_pruefsumme() -> str:
    """Prüfsumme der Quelltexte, die die vorgerechneten Reihen erzeugen (Zeilenenden normalisiert); `tools/messreihen.py` schreibt sie in die Datei, ein Test vergleicht."""
    h = hashlib.sha256()
    for name in QUELLEN:
        h.update((Path(__file__).resolve().parent / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]
