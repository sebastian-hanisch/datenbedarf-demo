"""Wasserstein-Abstand zweier Stichproben mit gleichen Gewichten (Zuordnung), Maßstab, Abstand zur Wahrheit und Zwei-Punkt-Formeln. Ohne Streamlit.

Zwei Stichproben X_1 ... X_n und Y_1 ... Y_n (Zeilen der Matrizen X und Y, Punkte im R^d) haben die empirischen Verteilungen mit Gewicht 1/n je Punkt. Ihr Abstand ist die
Zuordnung mit den kleinsten mittleren Kosten (jeder Punkt genau einem Partner): W1 mit euklidischen Entfernungen, W2² mit quadrierten. In einer Dimension ist die sortierte
Zuordnung die optimale. `scipy.optimize.linear_sum_assignment` ist zur Laufzeit erlaubt (Ausnahme dieses Stücks, nur diese eine Funktion); fehlt scipy, rechnet die eigene
Ungarische Methode (`dbd_zuordnung.py`) denselben Wert."""

import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np

from dbd_zuordnung import ungarisch

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:  # scipy fehlt: die eigene Zuordnung ist der Rückfall
    _lsa = None

_erf = np.vectorize(math.erf, otypes=[float])


def scipy_verfuegbar() -> bool:
    return _lsa is not None


def zuordnung(kosten: np.ndarray, verfahren: str = "auto") -> np.ndarray:
    """Spaltenindex je Zeile der kostenminimalen Zuordnung (quadratische Kostenmatrix). `verfahren`: "auto" (scipy, sonst eigene), "scipy" oder "eigen"."""
    kosten = np.asarray(kosten, dtype=float)
    if kosten.ndim != 2 or kosten.shape[0] != kosten.shape[1] or kosten.shape[0] == 0:
        raise ValueError("die Zuordnung braucht eine quadratische, nichtleere Kostenmatrix")
    if verfahren == "scipy" and _lsa is None:
        raise ImportError("scipy ist nicht installiert")
    if verfahren in ("scipy", "auto") and _lsa is not None:
        zeilen, spalten = _lsa(kosten)
        return np.asarray(spalten)[np.argsort(zeilen)]
    return ungarisch(kosten)[0]


def kosten_quadrat(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Matrix der quadrierten euklidischen Entfernungen, Eintrag (i, j) = |X_i − Y_j|² (Summe über die Achsen, ohne Auslöschung)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    d = np.zeros((X.shape[0], Y.shape[0]))
    for k in range(X.shape[1]):
        diff = X[:, k][:, None] - Y[:, k][None, :]
        d += diff * diff
    return d


def w1_zwei(X: np.ndarray, Y: np.ndarray, verfahren: str = "auto") -> float:
    """Zwei-Stichproben-W1: mittlere euklidische Entfernung der Paare der optimalen Zuordnung (d = 1: sortierte Zuordnung)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[1] == 1:
        return float(np.mean(np.abs(np.sort(X[:, 0]) - np.sort(Y[:, 0]))))
    c = np.sqrt(kosten_quadrat(X, Y))
    col = zuordnung(c, verfahren)
    return float(c[np.arange(c.shape[0]), col].mean())


def w2sq_zwei(X: np.ndarray, Y: np.ndarray, verfahren: str = "auto") -> float:
    """Zwei-Stichproben-W2² (mittlere quadrierte Entfernung der optimalen Zuordnung; d = 1: sortiert)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[1] == 1:
        return float(np.mean((np.sort(X[:, 0]) - np.sort(Y[:, 0])) ** 2))
    c = kosten_quadrat(X, Y)
    col = zuordnung(c, verfahren)
    return float(c[np.arange(c.shape[0]), col].mean())


def mittlerer_paarabstand(d: int) -> float:
    """Maßstab E‖X − X′‖ = 2 Γ((d+1)/2) / Γ(d/2) für zwei unabhängige Punkte aus N(0, I_d) (Standardbibliothek: lgamma)."""
    return 2.0 * math.exp(math.lgamma((d + 1) / 2.0) - math.lgamma(d / 2.0))


def normal_cdf(z) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def _g_stamm(z) -> np.ndarray:
    """G(z) = z Φ(z) + φ(z), Stammfunktion von Φ."""
    z = np.asarray(z, dtype=float)
    return z * normal_cdf(z) + np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def w1_wahrheit(x_sortiert: np.ndarray) -> np.ndarray:
    """Abstand zur Wahrheit bei d = 1: W1(P_n, N(0, 1)) = ∫ |F_n − Φ| exakt, stückweise geschlossen. Zeilen von `x_sortiert` (Form (R, n)) sind aufsteigende Stichproben."""
    x = np.atleast_2d(np.asarray(x_sortiert, dtype=float))
    n = x.shape[1]
    stamm = _g_stamm(x)                                              # ∫ Φ bis x
    links = stamm[:, 0]                                              # (−∞, x_1): F_n = 0, Integral von Φ
    rechts = _g_stamm(-x[:, -1])                                     # (x_n, ∞): F_n = 1, Integral von 1 − Φ
    if n == 1:
        return links + rechts
    c = np.arange(1, n) / n
    q = np.array([NormalDist().inv_cdf(float(ci)) for ci in c])      # Φ(q_i) = i/n: dort wechselt F_n − Φ das Vorzeichen
    lo, hi = x[:, :-1], x[:, 1:]
    xk = np.clip(q[None, :], lo, hi)
    s_k = _g_stamm(xk)
    innen = (c[None, :] * (xk - lo) - (s_k - stamm[:, :-1])) + ((stamm[:, 1:] - s_k) - c[None, :] * (hi - xk))
    return links + rechts + innen.sum(axis=1)


# --- Zwei-Punkt-Verteilung: exakte Formeln (Anteil der Einsen in zwei unabhängigen Stichproben der Größe N) ----------------------------------------

def zwei_punkt_zwei_stichproben(N: int) -> Fraction:
    """E|Z − Z′| = C(2N, N) / 4^N für die Anteile Z, Z′ der Einsen zweier unabhängiger Stichproben aus P = ½ δ_0 + ½ δ_1: der Zwei-Stichproben-W1."""
    return Fraction(math.comb(2 * N, N), 4 ** N)


def zwei_punkt_wahrheit(N: int) -> Fraction:
    """E|Z − ½| = C(N, N/2) / 2^(N+1) (N gerade): der Abstand der Stichprobe zur Wahrheit P."""
    if N % 2:
        raise ValueError("die Formel gilt für gerades N")
    return Fraction(math.comb(N, N // 2), 2 ** (N + 1))


def zwei_punkt_verhaeltnis(N: int) -> float:
    """Zwei-Stichproben-Abstand durch Abstand zur Wahrheit (exakt, N gerade): 3/2 bei N = 2, fallend gegen √2."""
    return float(zwei_punkt_zwei_stichproben(N) / zwei_punkt_wahrheit(N))


def steigung(ns, werte) -> float:
    """Steigung der Kleinste-Quadrate-Geraden von log(Werte) gegen log(n) (Potenzanpassung)."""
    return float(np.polyfit(np.log(np.asarray(ns, dtype=float)), np.log(np.asarray(werte, dtype=float)), 1)[0])


def potenzanpassung(ns, werte) -> tuple[float, float]:
    """(a, b) mit Werte ≈ a · n^b (Kleinste Quadrate in log-log)."""
    b, la = np.polyfit(np.log(np.asarray(ns, dtype=float)), np.log(np.asarray(werte, dtype=float)), 1)
    return float(math.exp(la)), float(b)
