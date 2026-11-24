"""Zufallsströme der Demo: vektorisierter SplitMix64 (nur numpy), Ströme je (Zufalls-Seed, Zweck, Zeile).

Die gleichverteilten Zahlen sind bitgleich zu `SplitMix64.random()` aus `dbd_rng.py` (Test `test_zufall.py`). Die Normalzahlen sind Box-Muller wie `SplitMix64.normal()`, aber mit
numpy-Funktionen statt der Dezimalrechnung: der Hausgenerator schafft rund 17 000 Normalzahlen je Sekunde (Zeitmessung wie in der README, je nach Rechner und Last verschieden), ein Durchlauf der Demo braucht Millionen. Dadurch sind die
Normalzahlen nur bis auf die letzte Binärstelle plattformgleich; alle Zahlen der Demo sind Mittel und Steigungen und werden als Band geprüft."""

import math

import numpy as np

from dbd_rng import MASK, SplitMix64

_GAMMA = 0x9E3779B97F4A7C15


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Zufalls-Seed, Zweck, Index); die ersten zwei Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def zufall_gleich(seed: int, zweck: int, zeilen: int, spalten: int, start: int = 0) -> np.ndarray:
    """Gleichverteilte Zahlen in [0, 1): `zeilen` eigene Ströme (Zeile j = Strom (seed, zweck, start + j)) mit je `spalten` Zahlen, dieselben wie `SplitMix64.random()`."""
    z0 = np.array([instanz_rng(seed, zweck, start + j).state for j in range(zeilen)], dtype=np.uint64)
    i = np.arange(1, spalten + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z0[:, None] + i[None, :] * np.uint64(_GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def zufall_normal(seed: int, zweck: int, zeilen: int, spalten: int, start: int = 0) -> np.ndarray:
    """Standardnormalzahlen (Form (zeilen, spalten)): Box-Muller aus zwei gleichverteilten Zahlen je Normalzahl, wie `SplitMix64.normal()`."""
    u = zufall_gleich(seed, zweck, zeilen, 2 * spalten, start)
    u1 = np.maximum(u[:, 0::2], 1e-12)
    u2 = u[:, 1::2]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


def stichproben(seed: int, zweck: int, anzahl: int, n: int, d: int, start: int = 0) -> np.ndarray:
    """`anzahl` Stichproben aus N(0, I_d) mit je n Punkten, Form (anzahl, n, d); Stichprobe j ist Strom (seed, zweck, start + j)."""
    return zufall_normal(seed, zweck, anzahl, n * d, start).reshape(anzahl, n, d)
