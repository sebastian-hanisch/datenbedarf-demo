"""Reihen, Stichproben-Familien, Minibatch und die Einheiten von `dbd_sliced.py` (kleine Instanzen, von Hand nachgerechnet)."""

import math

import numpy as np
import pytest

import dbd_sliced as SL
import dbd_sweeps as S
import dbd_wasserstein as W


# --- Familien ---------------------------------------------------------------------------------------------------------------------------------

def test_stichprobe_paar_hat_die_form_und_die_familien_die_richtige_struktur():
    x, y = S.stichprobe_paar("normal", 4, 0.0, 50, 1, 0)
    assert x.shape == y.shape == (50, 4)
    g, _ = S.stichprobe_paar("gerade", 10, 0.3, 4000, 1, 0)
    assert abs(g[:, 0].std() - 1.0) < 0.05 and abs(g[:, 1:].std() - 0.3) < 0.02
    b, _ = S.stichprobe_paar("bloecke", 10, 0.1, 4000, 1, 0)
    assert abs(np.mean(b[:, 0] > 0) - 0.5) < 0.03 and abs(np.abs(b[:, 0]).mean() - 3.0) < 0.05 and abs(b[:, 1:].std() - 0.1) < 0.01
    a, _ = S.stichprobe_paar("bloecke", 10, 0.0, 100, 1, 0)
    assert set(np.unique(a[:, 0])) == {-3.0, 3.0} and np.all(a[:, 1:] == 0.0)


def test_unbekannte_familie_wird_abgelehnt():
    with pytest.raises(ValueError):
        S.stichprobe_paar("kugel", 3, 0.0, 10, 1, 0)


def test_ns_bis_gibt_das_raster_der_verdopplungen():
    assert S.ns_bis(1024) == S.NS_ALLE == (32, 64, 128, 256, 512, 1024) and S.ns_bis(256) == (32, 64, 128, 256) and S.ns_bis(32) == (32,)


# --- Reihen ------------------------------------------------------------------------------------------------------------------------------------

def test_reihe_gibt_form_mittel_und_steigung_ueber_alle_und_die_letzten_drei():
    r = S.reihe("normal", 2, 0.0, (32, 64, 128, 256), 5, 7)
    assert r["w1"].shape == (4, 5) and np.allclose(r["mittel"], r["w1"].mean(axis=1))
    assert r["steigung_alle"] == pytest.approx(W.steigung(r["ns"], r["mittel"])) and r["steigung_letzte3"] == pytest.approx(W.steigung(r["ns"][-3:], r["mittel"][-3:]))
    assert math.isnan(S.reihe("normal", 2, 0.0, (32, 64), 3, 7)["steigung_letzte3"])


def test_reihe_normal_teilt_durch_den_massstab():
    r = S.reihe_normal(3, 128, 5, 7)
    assert r["massstab"] == pytest.approx(W.mittlerer_paarabstand(3)) and np.allclose(r["verhaeltnis"], r["mittel"] / r["massstab"])
    assert r["steigung_alle"] == pytest.approx(W.steigung(r["ns"], r["verhaeltnis"]))


def test_noetiges_n_von_hand_aus_einem_exakten_potenzgesetz():
    """W1/Maßstab = 2 n^(−1/2): 10 % bei n = (2/0,1)² = 400 (innerhalb des Rasters bis 1 024, kein Stern), 5 % bei n = 1 600 (Stern)."""
    ns = (32, 64, 128, 256, 512, 1024)
    v = [2.0 * n ** -0.5 for n in ns]
    n10, stern10 = S.noetiges_n(ns, v, 0.10)
    n5, stern5 = S.noetiges_n(ns, v, 0.05)
    assert n10 == pytest.approx(400.0, rel=1e-9) and not stern10
    assert n5 == pytest.approx(1600.0, rel=1e-9) and stern5


def test_zwei_gegen_wahrheit_gleicht_der_einzelrechnung_aus_den_stichproben():
    z = S.zwei_gegen_wahrheit(64, 5, 11)
    zwei, wahr = [], []
    for rep in range(5):
        x, y = S.stichprobe_paar("normal", 1, 0.0, 64, 11, rep)
        zwei.append(W.w1_zwei(x, y))
        wahr.append(W.w1_wahrheit(np.sort(x[:, 0]))[0])
    assert z["zwei"] == pytest.approx(np.mean(zwei), rel=1e-12) and z["wahrheit"] == pytest.approx(np.mean(wahr), rel=1e-12)
    assert z["verhaeltnis"] == pytest.approx(np.mean(zwei) / np.mean(wahr), rel=1e-12)


# --- Minibatch ---------------------------------------------------------------------------------------------------------------------------------

def test_minibatch_werte_sind_reproduzierbar_unabhaengig_vom_block_und_nehmen_den_zufalls_seed_ernst():
    a = S.minibatch_werte(2, 8, 30, 5, "verschoben")
    assert np.array_equal(a, S.minibatch_werte(2, 8, 30, 5, "verschoben")) and not np.array_equal(a, S.minibatch_werte(2, 8, 30, 6, "verschoben"))
    assert np.array_equal(a[:10], S.minibatch_werte(2, 8, 10, 5, "verschoben"))


def test_minibatch_k_gleich_1_ist_die_entfernung_zweier_punkte():
    """Bei k = 1 ist der Wert die quadrierte Entfernung der zwei Punkte; bei P = Q und d = 2 ist ihr Mittel 2d = 4 (Zufallspaarung), Band 4 Standardfehler."""
    w = S.minibatch_werte(2, 1, 4000, 3, "gleich")
    assert abs(w.mean() - 4.0) < 4 * w.std(ddof=1) / math.sqrt(len(w))


def test_wahres_w2_quadrat_des_minibatch_beispiels_von_hand():
    """N(0, I) gegen N(2 e₁, 1,5² I): |Δμ|² + d (s − 1)² = 4 + d · 0,25."""
    assert S.w2sq_minibatch_wahr(2, "verschoben") == 4.5 and S.w2sq_minibatch_wahr(10, "verschoben") == 6.5 and S.w2sq_minibatch_wahr(10, "gleich") == 0.0


def test_minibatch_reihe_enthaelt_v_anteil_und_zufallspaarung():
    r = S.minibatch_reihe(2, "verschoben", 100, 4, ks=(8, 32))
    assert [z["k"] for z in r["zeilen"]] == [8, 32] and math.isnan(r["zufallspaarung"])
    for z in r["zeilen"]:
        assert z["v"] == pytest.approx(z["mittel"] / 4.5) and 0.0 <= z["anteil_unter_1"] <= 1.0
    g = S.minibatch_reihe(10, "gleich", 50, 4, ks=(8,))
    assert g["zufallspaarung"] == 20.0 and math.isnan(g["zeilen"][0]["v"]) and math.isnan(g["zeilen"][0]["anteil_unter_1"])


def test_steigung_von_log_v_minus_1_von_hand():
    """V − 1 = 0,2 bei k = 32 und 0,05 bei k = 128: Steigung = ln(0,05/0,2)/ln(4) = −1."""
    zeilen = [{"k": 32, "v": 1.2}, {"k": 128, "v": 1.05}]
    assert S.steigung_v_minus_1(zeilen) == pytest.approx(-1.0, abs=1e-12)


def test_extrapolation_vergleicht_mit_der_vorhersage_der_anpassung():
    ns = (32, 64, 128)
    a, b = 0.9, -0.5
    z = S.extrapolation(1, ns, [a * n ** b for n in ns], ns=(256,), wdh=40, seed=3)
    assert z[0]["vorhergesagt"] == pytest.approx(a * 256 ** b, rel=1e-10) and z[0]["verhaeltnis"] == pytest.approx(z[0]["gemessen"] / z[0]["vorhergesagt"])
    assert z[0]["se"] > 0


# --- sliced -----------------------------------------------------------------------------------------------------------------------------------

def test_vierzig_faelle_in_vier_familien_zu_zehn_staerken():
    assert len(SL.faelle()) == 40 and SL.STAERKEN == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert [f for f, _ in SL.faelle()][::10] == ["S", "I", "E", "K"]


def test_kovarianzmatrizen_sind_symmetrisch_positiv_definit_mit_den_genannten_werten():
    for f in SL.FAMILIEN:
        for m in SL.STAERKEN:
            s = SL.sigma_matrix(f, m)
            assert np.allclose(s, s.T) and np.linalg.eigvalsh(s).min() > 0
    assert SL.sigma_matrix("I", 0.5)[0, 0] == pytest.approx(2.25) and SL.sigma_matrix("E", 0.5)[0, 0] == pytest.approx(6.25) and SL.sigma_matrix("E", 0.5)[1, 1] == 1.0
    k = SL.sigma_matrix("K", 0.4)
    assert k[0, 0] == pytest.approx(1.0) and k[0, 1] == pytest.approx(0.2)


def test_raenge_mit_mittleren_raengen_bei_gleichstand():
    assert list(SL.raenge([3.0, 1.0, 2.0])) == [3.0, 1.0, 2.0]
    assert list(SL.raenge([1.0, 2.0, 2.0, 3.0])) == [1.0, 2.5, 2.5, 4.0]
    assert list(SL.raenge([0.3, 0.1 + 0.2])) == [1.5, 1.5]            # gleich bis auf Rundung


def test_spearman_und_kendall_von_hand():
    a = [1, 2, 3, 4]
    assert SL.spearman(a, a) == pytest.approx(1.0) and SL.spearman(a, a[::-1]) == pytest.approx(-1.0)
    assert SL.kendall(a, a) == pytest.approx(1.0) and SL.kendall(a, a[::-1]) == pytest.approx(-1.0)
    # eine Vertauschung benachbarter Werte: 5 von 6 Paaren stimmen überein: τ = (5 − 1)/6
    assert SL.kendall([1, 2, 3, 4], [2, 1, 3, 4]) == pytest.approx(4 / 6)


def test_stichproben_fall_s_hat_die_verschiebung_der_laenge_m_und_die_anderen_familien_mittel_null():
    y = SL.stichproben_fall(2026, "S", 1.0, 0, n=6000)
    assert np.linalg.norm(y.mean(axis=0)) == pytest.approx(1.0, abs=0.06)
    e = SL.stichproben_fall(2026, "E", 0.5, 3, n=6000)
    assert abs(e[:, 0].std() - 2.5) < 0.08 and abs(e[:, 1:].std() - 1.0) < 0.02


def test_sliced_aus_stichproben_ist_null_bei_gleichen_stichproben_und_symmetrisch():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
    g = rng.normal(size=(20, 4))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    assert SL.sliced_aus_stichproben(x, x, g) == 0.0
    assert SL.sliced_aus_stichproben(x, y, g) == pytest.approx(SL.sliced_aus_stichproben(y, x, g))
