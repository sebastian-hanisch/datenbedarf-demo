"""Eigenschaftstests (metamorph): Aussagen, die für den Zwei-Stichproben-Abstand gelten müssen, ohne dass das richtige Ergebnis bekannt ist (je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed)."""

import numpy as np
import pytest

import dbd_sweeps as S
import dbd_wasserstein as W

N_INSTANZEN = 30


def _paare(seed, n=24, d=3, anzahl=N_INSTANZEN):
    rng = np.random.default_rng(np.random.SeedSequence([8810, seed]))
    return [(rng.normal(size=(n, d)), rng.normal(size=(n, d)) * 1.3 + 0.2) for _ in range(anzahl)]


def test_skalieren_beider_stichproben_skaliert_w1_mit_dem_faktor_und_w2_quadrat_mit_dem_quadrat():
    for (x, y), c in zip(_paare(1), np.linspace(0.1, 7.0, N_INSTANZEN)):
        assert W.w1_zwei(c * x, c * y) == pytest.approx(c * W.w1_zwei(x, y), rel=1e-10)
        assert W.w2sq_zwei(c * x, c * y) == pytest.approx(c * c * W.w2sq_zwei(x, y), rel=1e-10)


def test_gemeinsame_verschiebung_und_drehung_aendern_nichts():
    rng = np.random.default_rng(5)
    for x, y in _paare(2):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        t = rng.normal(size=3) * 4
        assert W.w1_zwei(x @ q + t, y @ q + t) == pytest.approx(W.w1_zwei(x, y), rel=1e-10)


def test_vertauschen_der_stichproben_und_umordnen_der_punkte_aendert_nichts():
    rng = np.random.default_rng(6)
    for x, y in _paare(3):
        w = W.w1_zwei(x, y)
        assert W.w1_zwei(y, x) == pytest.approx(w, rel=1e-12)
        assert W.w1_zwei(x[rng.permutation(len(x))], y[rng.permutation(len(y))]) == pytest.approx(w, rel=1e-12)


def test_w1_ist_hoechstens_die_wurzel_aus_w2_quadrat():
    for x, y in _paare(4):
        assert W.w1_zwei(x, y) <= np.sqrt(W.w2sq_zwei(x, y)) + 1e-12


def test_abstand_ist_nichtnegativ_und_null_nur_bei_gleichen_punktmengen():
    for x, y in _paare(5):
        assert W.w1_zwei(x, y) > 0.0
        assert W.w1_zwei(x, x) == pytest.approx(0.0, abs=1e-12)


def test_dreiecksungleichung_fuer_w1_der_stichproben():
    rng = np.random.default_rng(7)
    for _ in range(N_INSTANZEN):
        a, b, c = (rng.normal(size=(16, 2)) * s for s in (1.0, 1.5, 0.7))
        assert W.w1_zwei(a, c) <= W.w1_zwei(a, b) + W.w1_zwei(b, c) + 1e-12


def test_mehr_daten_senken_das_mittel_ueber_zufalls_seeds():
    """d = 3: Mittel über 20 Wiederholungen fällt von n = 16 über 64 auf 256 (Band: jede Stufe um mehr als 10 %)."""
    m = [S.w1_wiederholungen("normal", 3, 0.0, n, 20, 11).mean() for n in (16, 64, 256)]
    assert m[1] < 0.9 * m[0] and m[2] < 0.9 * m[1]


def test_punkte_auf_einer_geraden_gleichen_der_eindimensionalen_zuordnung_in_jeder_dimension():
    for d in (2, 5, 10):
        for rep in range(5):
            x, y = S.stichprobe_paar("gerade", d, 0.0, 80, 2026, rep)
            assert np.all(x[:, 1:] == 0.0) and np.all(y[:, 1:] == 0.0)
            assert W.w1_zwei(x, y) == pytest.approx(W.w1_zwei(x[:, :1], y[:, :1]), abs=1e-10)


def test_zwei_atome_haben_w1_gleich_abstand_mal_betrag_der_anteilsdifferenz_in_jeder_dimension():
    delta = 2 * S.BLOCK_ABSTAND
    for d in (2, 10):
        for rep in range(5):
            x, y = S.stichprobe_paar("bloecke", d, 0.0, 120, 2026, rep)
            anteil = lambda p: float(np.mean(p[:, 0] > 0))                                       # noqa: E731
            assert W.w1_zwei(x, y) == pytest.approx(delta * abs(anteil(x) - anteil(y)), abs=1e-10)


def test_gleicher_zufalls_seed_gleiche_reihe_und_mehr_wiederholungen_enthalten_die_ersten():
    a = S.w1_wiederholungen("normal", 4, 0.0, 32, 10, 99)
    assert np.array_equal(a, S.w1_wiederholungen("normal", 4, 0.0, 32, 10, 99))
    assert np.array_equal(a, S.w1_wiederholungen("normal", 4, 0.0, 32, 30, 99)[:10])
    assert not np.array_equal(a, S.w1_wiederholungen("normal", 4, 0.0, 32, 10, 100))


def test_stichproben_einer_wiederholung_sind_verschieden_und_die_reihe_haengt_nicht_am_n_raster():
    x, y = S.stichprobe_paar("normal", 3, 0.0, 64, 5, 0)
    assert not np.array_equal(x, y)
    a = S.reihe("normal", 3, 0.0, (32, 64), 4, 5)
    b = S.reihe("normal", 3, 0.0, (32, 64, 128), 4, 5)
    assert np.array_equal(a["w1"], b["w1"][:2])


def test_minibatch_mittel_liegt_ueber_dem_wahren_w2_quadrat_und_faellt_mit_k():
    """Verschoben und gestreckt, d = 2: Mittel über 300 Zufalls-Seeds ≥ wahres W2² = 4,5 für k = 8 und 32, fallend mit k (Band: 4 Standardfehler)."""
    r = S.minibatch_reihe(2, "verschoben", 300, 3, ks=(8, 32))
    for z in r["zeilen"]:
        assert z["mittel"] > r["wahr"] + 4 * z["se"]
    assert r["zeilen"][1]["mittel"] < r["zeilen"][0]["mittel"]


def test_minibatch_bei_gleichen_verteilungen_liegt_deutlich_unter_der_zufaelligen_paarung():
    r = S.minibatch_reihe(10, "gleich", 100, 3, ks=(8, 64))
    assert r["zufallspaarung"] == 20.0
    assert r["zeilen"][0]["mittel"] < 0.8 * 20 and r["zeilen"][1]["mittel"] < 0.5 * 20
    assert r["zeilen"][1]["mittel"] < r["zeilen"][0]["mittel"]
