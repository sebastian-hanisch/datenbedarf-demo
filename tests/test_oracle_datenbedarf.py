"""Orakel: Zuordnung, Abstand zur Wahrheit, Maßstab, sliced und Minibatch gegen unabhängige Rechenwege (Linearprogramm, Netzwerksimplex, Aufzählung, Quadratur, scipy.stats).

Jeder Test läuft unter einer Sekunde bis wenigen Sekunden; die Orakel selbst liegen nur hier (nicht in der Laufzeit)."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import dbd_sliced as SL
import dbd_sweeps as S
import dbd_wasserstein as W
import dbd_zufall as Z

scipy_opt = pytest.importorskip("scipy.optimize")
scipy_int = pytest.importorskip("scipy.integrate")
scipy_stats = pytest.importorskip("scipy.stats")
scipy_special = pytest.importorskip("scipy.special")
from scipy.spatial.distance import cdist  # noqa: E402


def _lp_wert(c: np.ndarray) -> float:
    """Transportproblem mit gleichen Gewichten 1/n als Linearprogramm (HiGHS): min Σ c_ij P_ij, Zeilen- und Spaltensummen 1/n."""
    n = c.shape[0]
    a = np.zeros((2 * n, n * n))
    for i in range(n):
        a[i, i * n:(i + 1) * n] = 1.0
        a[n + i, i::n] = 1.0
    res = scipy_opt.linprog(c.ravel(), A_eq=a, b_eq=np.full(2 * n, 1.0 / n), bounds=(0, None), method="highs")
    assert res.status == 0
    return float(res.fun)


def _aufzaehlung(c: np.ndarray) -> float:
    n = c.shape[0]
    return min(sum(c[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n))) / n


def test_zuordnung_gleich_linearprogramm_auf_15_zufallsinstanzen():
    rng = np.random.default_rng(np.random.SeedSequence([8801]))
    for i in range(15):
        n, d = int(rng.integers(3, 11)), int(rng.integers(2, 6))
        x, y = rng.normal(size=(n, d)), rng.normal(size=(n, d))
        assert W.w1_zwei(x, y) == pytest.approx(_lp_wert(cdist(x, y)), abs=1e-9)
        assert W.w2sq_zwei(x, y) == pytest.approx(_lp_wert(cdist(x, y, "sqeuclidean")), abs=1e-9)


def test_eigene_zuordnung_gleich_linearprogramm_und_aufzaehlung():
    rng = np.random.default_rng(np.random.SeedSequence([8802]))
    for i in range(20):
        n = int(rng.integers(2, 8))
        x, y = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        c = cdist(x, y)
        assert W.w1_zwei(x, y, "eigen") == pytest.approx(_aufzaehlung(c), abs=1e-12)
    for i in range(5):
        x, y = rng.normal(size=(9, 3)), rng.normal(size=(9, 3))
        assert W.w1_zwei(x, y, "eigen") == pytest.approx(_lp_wert(cdist(x, y)), abs=1e-9)


def test_zuordnung_gleich_netzwerksimplex_von_networkx():
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(np.random.SeedSequence([8803]))
    for i in range(5):
        n = 8
        x, y = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
        c = np.round(cdist(x, y) * 1e6).astype(int)
        g = nx.DiGraph()
        for a in range(n):
            g.add_node(("x", a), demand=-1)
            g.add_node(("y", a), demand=1)
        for a in range(n):
            for b in range(n):
                g.add_edge(("x", a), ("y", b), weight=int(c[a, b]), capacity=1)
        kosten, _ = nx.network_simplex(g)
        assert W.w1_zwei(x, y) == pytest.approx(kosten / 1e6 / n, abs=1e-6)


def test_eindimensional_gleich_scipy_wasserstein_distance_und_quantilformel():
    rng = np.random.default_rng(np.random.SeedSequence([8804]))
    for i in range(20):
        n = int(rng.integers(5, 80))
        x, y = rng.normal(size=n), rng.normal(size=n) * 1.5 + 0.3
        assert W.w1_zwei(x[:, None], y[:, None]) == pytest.approx(scipy_stats.wasserstein_distance(x, y), abs=1e-12)
        assert W.w2sq_zwei(x[:, None], y[:, None]) == pytest.approx(np.mean((np.sort(x) - np.sort(y)) ** 2), abs=1e-14)


def test_massstab_gleich_gammaln_formel_und_monte_carlo():
    for d in (1, 2, 3, 5, 10):
        assert W.mittlerer_paarabstand(d) == pytest.approx(2 * math.exp(scipy_special.gammaln((d + 1) / 2) - scipy_special.gammaln(d / 2)), rel=1e-12)
    for d in (2, 10):
        x = Z.stichproben(77, 3, 1, 20000, d)[0]
        y = Z.stichproben(77, 4, 1, 20000, d)[0]
        mc = float(np.mean(np.linalg.norm(x - y, axis=1)))
        assert mc == pytest.approx(W.mittlerer_paarabstand(d), rel=0.01)


def test_abstand_zur_wahrheit_gleich_quadratur_und_trapezsumme():
    for n, seed in ((5, 1), (37, 2), (200, 3)):
        x = np.sort(Z.zufall_normal(seed, 90, 1, n)[0])
        exakt = float(W.w1_wahrheit(x)[0])
        f = lambda t: abs(np.searchsorted(x, t, side="right") / n - scipy_stats.norm.cdf(t))      # noqa: E731
        quad = scipy_int.quad(f, -12, 12, points=list(x), limit=2000, epsabs=1e-11)[0]
        gitter = np.linspace(-12, 12, 2_400_001)
        fg = np.abs(np.searchsorted(x, gitter, side="right") / n - scipy_stats.norm.cdf(gitter))
        trapez = float(np.sum((fg[1:] + fg[:-1]) * 0.5 * np.diff(gitter)))
        assert exakt == pytest.approx(quad, abs=1e-8)
        assert exakt == pytest.approx(trapez, abs=1e-5)


def test_abstand_zur_wahrheit_einer_einzelnen_beobachtung():
    """n = 1, x = 0: ∫_{-∞}^0 Φ + ∫_0^∞ (1 − Φ) = 2 φ(0) = 2/√(2π)."""
    assert W.w1_wahrheit(np.array([[0.0]]))[0] == pytest.approx(2 / math.sqrt(2 * math.pi), abs=1e-12)


def test_zwei_punkt_formeln_gleich_zwei_stichproben_abstand_der_aufzaehlung():
    """Alle Stichproben der Größe k aus {0, 1} (Anteil a der Einsen) durch die Funktion der Demo: Erwartung gleich C(2k, k)/4^k."""
    for k in range(1, 7):
        gesamt = F(0)
        for a in range(k + 1):
            for b in range(k + 1):
                x = np.array([[0.0]] * (k - a) + [[1.0]] * a)
                y = np.array([[0.0]] * (k - b) + [[1.0]] * b)
                gesamt += F(math.comb(k, a) * math.comb(k, b), 4 ** k) * F(W.w1_zwei(x, y)).limit_denominator(10 ** 6)
        assert float(gesamt) == pytest.approx(float(W.zwei_punkt_zwei_stichproben(k)), abs=1e-9)


def test_minibatch_zwei_punkt_beispiel_exakt_in_bruechen():
    """Q = P + 2 mit P = ½ δ0 + ½ δ1 (W2² = 4): Erwartung des Minibatch-Werts 9/2 bei k = 1 und 35/8 bei k = 2, V = 9/8 und 35/32; Basislinie bei P = Q: 1/2 und 3/8."""
    def wert(k, shift):
        gesamt = F(0)
        for a in range(k + 1):
            for b in range(k + 1):
                x = np.array([[0.0]] * (k - a) + [[1.0]] * a)
                y = np.array([[shift]] * (k - b) + [[shift + 1.0]] * b)
                gesamt += F(math.comb(k, a) * math.comb(k, b), 4 ** k) * F(W.w2sq_zwei(x, y)).limit_denominator(10 ** 6)
        return gesamt
    assert wert(1, 2) == F(9, 2) and wert(2, 2) == F(35, 8)
    assert wert(1, 2) / 4 == F(9, 8) and wert(2, 2) / 4 == F(35, 32)
    assert wert(1, 0) == F(1, 2) and wert(2, 0) == F(3, 8)


def test_minibatch_basislinie_gleich_zwei_stichproben_abstand_im_mittel():
    """P = Q: der Minibatch-Wert ist der Zwei-Stichproben-W2² mit n = k (andere Ströme, Mittel über 300 Wiederholungen, Band 4 Standardfehler)."""
    for d, k in ((2, 16), (10, 32)):
        mb = S.minibatch_werte(d, k, 300, 5, "gleich")
        zw = np.array([W.w2sq_zwei(*S.stichprobe_paar("normal", d, 0.0, k, 5, r)) for r in range(300)])
        se = math.sqrt(mb.var(ddof=1) / 300 + zw.var(ddof=1) / 300)
        assert abs(mb.mean() - zw.mean()) < 4 * se


# --- sliced ----------------------------------------------------------------------------------------------------------------------------------

def test_kugelerwartung_gleich_scipy_quad():
    """Glatte Integranden, Dimension 10 und 12 (die Dichte (1 − x²)^((d−3)/2) ist dort glatt genug für 300 Knoten; bei d = 4 liegt der Fehler bei 2e-8, bei d = 2 ist die Dichte singulär); die Demo braucht d = 10."""
    dichte = lambda x, d: (1 - x * x) ** ((d - 3) / 2)             # noqa: E731
    for d, f in ((10, lambda x: x * x), (10, lambda x: math.sqrt(1 + 15 * x * x) - 1), (12, lambda x: x ** 4), (12, lambda x: math.exp(x))):
        z = scipy_int.quad(lambda x: dichte(x, d), -1, 1, epsabs=1e-13, epsrel=1e-13)[0]
        zaehler = scipy_int.quad(lambda x: f(x) * dichte(x, d), -1, 1, epsabs=1e-13, epsrel=1e-13)[0]
        assert SL.e_kugel(np.vectorize(f), d) == pytest.approx(zaehler / z, abs=1e-9)


def test_sliced_population_von_hand_fuer_verschiebung_und_isotrope_streuung():
    """E (θ·u)² = 1/d: SW2 = m/√10 bei Verschiebung um m (W2 = m); isotrop: SW2 = m, W2 = m √10; beide Verhältnisse 1/√10."""
    for m in SL.STAERKEN:
        assert SL.sliced_population("S", m) == pytest.approx(m / math.sqrt(10), rel=1e-10)
        assert SL.bures("S", m) == pytest.approx(m, rel=1e-12)
        assert SL.sliced_population("I", m) == pytest.approx(m, rel=1e-12)
        assert SL.bures("I", m) == pytest.approx(m * math.sqrt(10), rel=1e-12)


def test_bures_gleich_gelbrich_formel_mit_matrixwurzel():
    linalg = pytest.importorskip("scipy.linalg")
    for f in ("I", "E", "K"):
        for m in (0.1, 0.5, 1.0):
            s = SL.sigma_matrix(f, m)
            w2sq = float(np.trace(np.eye(10) + s - 2 * linalg.sqrtm(s)).real)
            assert SL.bures(f, m) ** 2 == pytest.approx(w2sq, abs=1e-9)


def test_sliced_population_gleich_dem_mittel_ueber_viele_richtungen():
    """Familie E bei m = 0,5: Mittel von (√(θᵀΣθ) − 1)² über 20 000 Richtungen gegen die Quadratur (Band 2 %)."""
    g = Z.zufall_normal(3, 99, 1, 20000 * 10)[0].reshape(20000, 10)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    s = SL.sigma_matrix("E", 0.5)
    mc = math.sqrt(float(np.mean((np.sqrt(np.einsum("ld,de,le->l", g, s, g)) - 1.0) ** 2)))
    assert mc == pytest.approx(SL.sliced_population("E", 0.5), rel=0.02)


def test_sliced_aus_stichproben_naehert_sich_der_population():
    """Eine Verschiebung um 1 in Richtung e1: Stichproben n = 2000, 400 Richtungen gegen die Quadratur SW2 = 1/√10 (Band 5 %)."""
    x = Z.stichproben(5, 61, 1, 2000, 10)[0]
    y = Z.stichproben(5, 62, 1, 2000, 10)[0]
    y[:, 0] += 1.0
    g = Z.zufall_normal(5, 63, 1, 400 * 10)[0].reshape(400, 10)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    assert SL.sliced_aus_stichproben(x, y, g) == pytest.approx(1 / math.sqrt(10), rel=0.05)


def test_spearman_und_kendall_gleich_scipy_auch_bei_gleichstaenden():
    rng = np.random.default_rng(9)
    for _ in range(10):
        a = np.round(rng.normal(size=40), 1)
        b = np.round(a + rng.normal(size=40), 1)
        assert SL.spearman(a, b) == pytest.approx(scipy_stats.spearmanr(a, b).statistic, abs=1e-12)
        assert SL.kendall(a, b) == pytest.approx(scipy_stats.kendalltau(a, b).statistic, abs=1e-12)
