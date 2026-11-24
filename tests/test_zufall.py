"""Zufall: der vektorisierte Strom gleicht dem Hausgenerator (bitgleich bei den Gleichverteilten, bis auf die letzte Stelle bei den Normalzahlen)."""

import subprocess
import sys
from pathlib import Path

import numpy as np

import dbd_zufall as Z
from dbd_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent


def test_gleichverteilte_sind_bitgleich_zum_hausgenerator():
    u = Z.zufall_gleich(2026, 7, 3, 25, start=4)
    for zeile in range(3):
        g = Z.instanz_rng(2026, 7, 4 + zeile)
        assert np.array_equal(u[zeile], [g.random() for _ in range(25)])


def test_normalzahlen_gleichen_dem_hausgenerator_bis_auf_die_letzte_stelle():
    nz = Z.zufall_normal(11, 3, 2, 40, start=1)
    for zeile in range(2):
        g = Z.instanz_rng(11, 3, 1 + zeile)
        exakt = np.array([g.normal() for _ in range(40)])
        assert np.max(np.abs(nz[zeile] - exakt)) < 1e-12


def test_ein_strom_haengt_nicht_an_der_zahl_der_angeforderten_zeilen():
    a = Z.zufall_normal(5, 2, 6, 17)
    b = Z.zufall_normal(5, 2, 2, 17, start=3)
    assert np.array_equal(a[3:5], b)


def test_stichproben_haben_die_form_und_sind_reproduzierbar_und_verschieden():
    x = Z.stichproben(3, 9, 4, 20, 5)
    assert x.shape == (4, 20, 5)
    assert np.array_equal(x, Z.stichproben(3, 9, 4, 20, 5))
    assert not np.array_equal(x[0], x[1]) and not np.array_equal(x, Z.stichproben(4, 9, 4, 20, 5))


def test_normalzahlen_haben_mittel_null_und_varianz_eins():
    x = Z.zufall_normal(1, 1, 50, 2000)
    assert abs(x.mean()) < 0.01 and abs(x.var() - 1.0) < 0.02
    assert abs(np.mean(np.abs(x) < 1.0) - 0.6827) < 0.01


def test_zweiter_prozess_mit_anderem_hash_seed_rechnet_dasselbe():
    code = "import sys; sys.path.insert(0, sys.argv[1]); import dbd_sweeps as S; print(repr(S.reihe('normal', 3, 0.0, (32, 64), 4, 2026)['mittel'].tolist()))"
    ergebnisse = []
    for hs in ("1", "2"):
        r = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONHASHSEED": hs}, timeout=120)
        assert r.returncode == 0, r.stderr
        ergebnisse.append(r.stdout)
    assert ergebnisse[0] == ergebnisse[1] and ergebnisse[0].strip()
