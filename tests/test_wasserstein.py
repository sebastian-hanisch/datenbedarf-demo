"""Einheiten von `dbd_wasserstein.py` gegen Handrechnungen (Handbeispiele 1 bis 4 des Plans, in Brüchen), Randfälle und der Rückfall ohne scipy."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import dbd_wasserstein as W


# --- Handbeispiel 1: d = 1, drei Werte -----------------------------------------------------------------------------------------------------

X1 = np.array([[0.0], [1.0], [4.0]])
Y1 = np.array([[0.0], [2.0], [2.0]])


def test_handbeispiel_drei_werte_sortierte_zuordnung():
    """x = (0, 1, 4), y = (0, 2, 2): sortiert 0-0, 1-2, 4-2, W1 = (0 + 1 + 2)/3 = 1, W2² = (0 + 1 + 4)/3 = 5/3."""
    assert W.w1_zwei(X1, Y1) == pytest.approx((0 + 1 + 2) / 3, abs=1e-15)
    assert W.w2sq_zwei(X1, Y1) == pytest.approx((0 + 1 + 4) / 3, abs=1e-15)


def test_handbeispiel_drei_werte_allgemeine_zuordnung_gibt_dasselbe():
    """Dieselben Punkte mit einer zweiten, ruhenden Achse: der Pfad der allgemeinen Zuordnung (nicht sortiert) rechnet denselben Wert."""
    x2 = np.hstack([X1, np.zeros((3, 1))])
    y2 = np.hstack([Y1, np.zeros((3, 1))])
    assert W.w1_zwei(x2, y2) == pytest.approx(1.0, abs=1e-12)
    assert W.w2sq_zwei(x2, y2) == pytest.approx(5 / 3, abs=1e-12)


def test_handbeispiel_drei_werte_gleich_der_aufzaehlung_aller_sechs_zuordnungen():
    kosten = [sum(abs(float(X1[i, 0] - Y1[p[i], 0])) for i in range(3)) / 3 for p in itertools.permutations(range(3))]
    assert min(kosten) == pytest.approx(1.0, abs=1e-15)
    assert W.w1_zwei(X1, Y1) == pytest.approx(min(kosten), abs=1e-15)


def test_handbeispiel_drei_werte_gleich_dem_integral_der_verteilungsfunktionen():
    """F springt bei 0, 1, 4 um je 1/3, G bei 0 um 1/3 und bei 2 um 2/3: |F − G| = 0 auf [0, 1), 1/3 auf [1, 2) und 1/3 auf [2, 4); das Integral ist 1/3 · 1 + 1/3 · 2 = 1."""
    integral = F(0) * 1 + F(1, 3) * 1 + F(1, 3) * 2
    assert integral == 1
    assert W.w1_zwei(X1, Y1) == pytest.approx(float(integral), abs=1e-15)


# --- Handbeispiel 2: Zwei-Punkt-Verteilung -----------------------------------------------------------------------------------------------------

def _binom_pmf(n):
    return [F(math.comb(n, k), 2 ** n) for k in range(n + 1)]


def _e_zwei(N):
    pm = _binom_pmf(N)
    return sum(pm[a] * pm[b] * abs(F(a - b, N)) for a in range(N + 1) for b in range(N + 1))


def _e_wahr(N):
    pm = _binom_pmf(N)
    return sum(pm[a] * abs(F(a, N) - F(1, 2)) for a in range(N + 1))


def test_zwei_punkt_von_hand_bei_n_gleich_2_und_4():
    assert W.zwei_punkt_zwei_stichproben(2) == F(3, 8) and W.zwei_punkt_wahrheit(2) == F(1, 4)
    assert W.zwei_punkt_zwei_stichproben(4) == F(35, 128) and W.zwei_punkt_wahrheit(4) == F(3, 16)
    assert W.zwei_punkt_verhaeltnis(2) == 1.5
    assert W.zwei_punkt_verhaeltnis(4) == pytest.approx(float(F(35, 128) / F(3, 16)), abs=1e-15)         # 35/24


@pytest.mark.parametrize("N", [2, 4, 6, 8, 10, 12])
def test_zwei_punkt_formeln_gleich_der_aufzaehlung_aller_stichproben(N):
    assert W.zwei_punkt_zwei_stichproben(N) == _e_zwei(N)
    assert W.zwei_punkt_wahrheit(N) == _e_wahr(N)


def test_zwei_punkt_verhaeltnis_faellt_gegen_wurzel_zwei_und_hat_die_steigung_minus_ein_halb():
    ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    v = [W.zwei_punkt_verhaeltnis(N) for N in ns]
    assert all(a > b for a, b in zip(v, v[1:]))
    assert abs(v[-1] - math.sqrt(2)) < 1e-3 and abs(W.zwei_punkt_verhaeltnis(32) - 1.4197) < 5e-4
    werte = [float(W.zwei_punkt_zwei_stichproben(N)) for N in ns[4:]]
    assert W.steigung(ns[4:], werte) == pytest.approx(-0.5, abs=0.01)
    assert math.sqrt(1024) * float(W.zwei_punkt_zwei_stichproben(1024)) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-3)


def test_zwei_punkt_wahrheit_braucht_gerades_n():
    with pytest.raises(ValueError):
        W.zwei_punkt_wahrheit(3)


# --- Maßstab ----------------------------------------------------------------------------------------------------------------------------------

def test_massstab_von_hand_fuer_d_gleich_1_2_3():
    """2 Γ((d+1)/2)/Γ(d/2): d = 1: 2 Γ(1)/Γ(1/2) = 2/√π; d = 2: 2 Γ(3/2)/Γ(1) = √π; d = 3: 2 Γ(2)/Γ(3/2) = 4/√π."""
    assert W.mittlerer_paarabstand(1) == pytest.approx(2 / math.sqrt(math.pi), rel=1e-13)
    assert W.mittlerer_paarabstand(2) == pytest.approx(math.sqrt(math.pi), rel=1e-13)
    assert W.mittlerer_paarabstand(3) == pytest.approx(4 / math.sqrt(math.pi), rel=1e-13)


# --- Handbeispiel 3 und 4: Eigenschaften der Zuordnung -------------------------------------------------------------------------------------------

def test_zwei_atome_in_r10_haben_w1_gleich_delta_mal_betrag_der_anteilsdifferenz():
    rng = np.random.default_rng(np.random.SeedSequence([9800, 1]))
    n, d, delta = 400, 10, 6.0
    for _ in range(5):
        zx, zy = rng.random(n) < 0.5, rng.random(n) < 0.5
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        X[:, 0], Y[:, 0] = np.where(zx, delta, 0.0), np.where(zy, delta, 0.0)
        assert W.w1_zwei(X, Y) == pytest.approx(delta * abs(zx.mean() - zy.mean()), abs=1e-12)


def test_punkte_auf_einer_geraden_in_r10_gleichen_der_sortierten_zuordnung():
    rng = np.random.default_rng(np.random.SeedSequence([9800, 2]))
    u = rng.normal(size=10)
    u /= np.linalg.norm(u)
    for _ in range(5):
        t1, t2 = rng.normal(size=200), rng.normal(size=200)
        assert W.w1_zwei(np.outer(t1, u), np.outer(t2, u)) == pytest.approx(np.mean(np.abs(np.sort(t1) - np.sort(t2))), abs=1e-10)


# --- Randfälle ------------------------------------------------------------------------------------------------------------------------------

def test_n_gleich_1_ist_die_entfernung_der_zwei_punkte():
    assert W.w1_zwei(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])) == pytest.approx(5.0)
    assert W.w2sq_zwei(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])) == pytest.approx(25.0)
    assert W.w1_zwei(np.array([[2.0]]), np.array([[-1.0]])) == pytest.approx(3.0)


def test_n_gleich_2_waehlt_die_bessere_der_zwei_zuordnungen():
    x = np.array([[0.0, 0.0], [10.0, 0.0]])
    y = np.array([[10.0, 1.0], [0.0, 1.0]])               # kreuzende Zuordnung kostet 2 · 10, die richtige 2 · 1
    assert W.w1_zwei(x, y) == pytest.approx(1.0)


def test_gleiche_stichproben_haben_abstand_null_auch_bei_vertauschter_reihenfolge():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 4))
    assert W.w1_zwei(x, x) == pytest.approx(0.0, abs=1e-12)
    assert W.w1_zwei(x, x[::-1]) == pytest.approx(0.0, abs=1e-12)
    assert W.w2sq_zwei(x, x[rng.permutation(30)]) == pytest.approx(0.0, abs=1e-12)


def test_identische_punkte_gleichstand_der_zuordnung_hat_denselben_wert():
    x = np.zeros((6, 3))
    y = np.vstack([np.zeros((3, 3)), np.ones((3, 3))])
    assert W.w1_zwei(x, y) == pytest.approx(0.5 * math.sqrt(3))


def test_zuordnung_prueft_die_form_der_kostenmatrix():
    with pytest.raises(ValueError):
        W.zuordnung(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        W.zuordnung(np.zeros((0, 0)))


def test_zuordnung_ist_eine_permutation():
    rng = np.random.default_rng(2)
    for verfahren in ("auto", "eigen"):
        col = W.zuordnung(rng.random((12, 12)), verfahren)
        assert sorted(col) == list(range(12))


# --- Rückfall ohne scipy ---------------------------------------------------------------------------------------------------------------------

def test_rueckfall_ohne_scipy_rechnet_denselben_wert(monkeypatch):
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(60, 5)), rng.normal(size=(60, 5))
    mit = W.w1_zwei(x, y)
    monkeypatch.setattr(W, "_lsa", None)
    assert not W.scipy_verfuegbar()
    assert W.w1_zwei(x, y) == pytest.approx(mit, rel=1e-12)
    assert W.w2sq_zwei(x, y) == pytest.approx(W.w2sq_zwei(x, y, "eigen"), rel=1e-12)
    with pytest.raises(ImportError):
        W.zuordnung(np.eye(3), "scipy")


def test_eigene_zuordnung_gleicht_scipy_auf_zufallsinstanzen():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(4)
    for n in (3, 8, 20, 50):
        for _ in range(5):
            c = rng.random((n, n))
            a = W.zuordnung(c, "scipy")
            b = W.zuordnung(c, "eigen")
            assert c[np.arange(n), a].sum() == pytest.approx(c[np.arange(n), b].sum(), abs=1e-10)


# --- Steigung und Anpassung -----------------------------------------------------------------------------------------------------------------

def test_steigung_und_potenzanpassung_finden_ein_exaktes_potenzgesetz():
    ns = np.array([32, 64, 128, 256])
    werte = 3.0 * ns ** -0.4
    assert W.steigung(ns, werte) == pytest.approx(-0.4, abs=1e-12)
    a, b = W.potenzanpassung(ns, werte)
    assert a == pytest.approx(3.0, rel=1e-10) and b == pytest.approx(-0.4, abs=1e-12)
