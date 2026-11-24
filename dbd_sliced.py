"""Geschnittener Wasserstein-Abstand (sliced) gegen den Wasserstein-Abstand zwischen Gauß-Verteilungen: vier Familien, zehn Stärken, d = 10. Ohne Streamlit.

Die Referenz ist N(0, I_10). Verglichen werden 40 Gauß-Verteilungen N(μ, Σ) in vier Familien zu je zehn Stärken m = 0,1; 0,2; ...; 1,0:
  S  Verschiebung um m in einer Richtung (|μ| = m, Σ = I);
  I  isotrope Streuung, Standardabweichung 1 + m in allen Achsen;
  E  Streuung nur in der ersten Achse, Standardabweichung 1 + 3m, die übrigen Achsen unverändert;
  K  Gleichkorrelation, alle Varianzen 1, alle Kovarianzen ρ = m/2.
Wahrheit ist der Wasserstein-Abstand W2 zwischen den Gauß-Verteilungen (geschlossen, Bures): W2² = |μ|² + Σ_i (1 − √λ_i)² mit den Eigenwerten λ_i von Σ.
Der geschnittene Abstand (Population) mittelt die eindimensionalen Abstände der Projektionen über alle Richtungen θ der Einheitskugel:
SW2² = E_θ[(θ·μ)² + (√(θᵀΣθ) − 1)²]; für diese Familien hängt θᵀΣθ nur von x = θ·u (u eine feste Richtung) ab, und x hat die Dichte ∝ (1 − x²)^((d−3)/2)."""

import math

import numpy as np

import dbd_wasserstein as W
import dbd_zufall as Z

D = 10
N_STICHPROBE = 300
STAERKEN = tuple(round(0.1 * i, 1) for i in range(1, 11))
FAMILIEN = ("S", "I", "E", "K")
FAMILIEN_NAMEN = {"S": "Verschiebung", "I": "isotrope Streuung", "E": "Streuung in einer Achse", "K": "Gleichkorrelation"}
L_WERTE = (1, 5, 20, 100, 500)
ZWECK_REF, ZWECK_Y, ZWECK_RICHTUNG, ZWECK_U = 71, 72, 73, 74

_KNOTEN, _GEWICHTE = np.polynomial.legendre.leggauss(300)


def e_kugel(f, d: int = D) -> float:
    """Erwartung von f(x), x = θ·u für θ gleichverteilt auf der Kugel S^(d−1): Dichte ∝ (1 − x²)^((d−3)/2), Gauß-Legendre mit 300 Knoten (glatte Integranden, Dimension ab etwa 8; bei kleinem d ist die Dichte an den Rändern nicht glatt)."""
    dichte = (1.0 - _KNOTEN ** 2) ** ((d - 3) / 2.0)
    return float(np.sum(_GEWICHTE * f(_KNOTEN) * dichte) / np.sum(_GEWICHTE * dichte))


def sigma_matrix(familie: str, m: float, d: int = D) -> np.ndarray:
    """Kovarianzmatrix Σ der Familie bei Stärke m."""
    if familie in ("S",):
        return np.eye(d)
    if familie == "I":
        return (1.0 + m) ** 2 * np.eye(d)
    if familie == "E":
        s = np.eye(d)
        s[0, 0] = (1.0 + 3.0 * m) ** 2
        return s
    if familie == "K":
        rho = m / 2.0
        return (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
    raise ValueError(familie)


def bures(familie: str, m: float, d: int = D) -> float:
    """W2 zwischen N(0, I_d) und N(μ, Σ) der Familie (geschlossen): √(|μ|² + Σ_i (1 − √λ_i)²)."""
    mu2 = m * m if familie == "S" else 0.0
    lam = np.linalg.eigvalsh(sigma_matrix(familie, m, d))
    return math.sqrt(mu2 + float(np.sum((1.0 - np.sqrt(lam)) ** 2)))


def sliced_population(familie: str, m: float, d: int = D) -> float:
    """SW2 der Population (Wurzel von E_θ[(θ·μ)² + (√(θᵀΣθ) − 1)²]) mit eindimensionaler Quadratur."""
    if familie == "S":
        sl2 = e_kugel(lambda x: (m * x) ** 2, d)
    elif familie == "I":
        sl2 = (1.0 + m - 1.0) ** 2
    elif familie == "E":
        lam = (1.0 + 3.0 * m) ** 2
        sl2 = e_kugel(lambda x: (np.sqrt(1.0 + (lam - 1.0) * x * x) - 1.0) ** 2, d)
    elif familie == "K":
        rho = m / 2.0
        sl2 = e_kugel(lambda x: (np.sqrt((1.0 - rho) + rho * d * x * x) - 1.0) ** 2, d)
    else:
        raise ValueError(familie)
    return math.sqrt(sl2)


def faelle() -> list[tuple[str, float]]:
    """Die 40 Fälle: Familie, Stärke."""
    return [(f, m) for f in FAMILIEN for m in STAERKEN]


def verhaeltnisse() -> dict[str, list[float]]:
    """Verhältnis SW2 / W2 (Population, geschlossen) je Familie und Stärke."""
    return {f: [sliced_population(f, m) / bures(f, m) for m in STAERKEN] for f in FAMILIEN}


def raenge(a) -> np.ndarray:
    """Ränge (1 = kleinster) mit mittleren Rängen bei Gleichstand; Werte, die sich erst jenseits der zwölften Stelle unterscheiden, gelten als gleich."""
    a = np.round(np.asarray(a, dtype=float), 12)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b) -> float:
    """Spearmans Rangkorrelation (Pearson-Korrelation der mittleren Ränge)."""
    return float(np.corrcoef(raenge(a), raenge(b))[0, 1])


def kendall(a, b) -> float:
    """Kendalls τ-b (mit Korrektur für Gleichstände)."""
    a = np.round(np.asarray(a, dtype=float), 12)
    b = np.round(np.asarray(b, dtype=float), 12)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(len(a), k=1)
    s = float(np.sum(da[iu] * db[iu]))
    na, nb = float(np.sum(da[iu] != 0)), float(np.sum(db[iu] != 0))
    return s / math.sqrt(na * nb)


def mittel_je_familie(werte, funktion=spearman, bezug=None) -> float:
    """Mittel über die vier Familien der Rangkorrelation innerhalb der Familie (zehn Stärken je Familie)."""
    werte = np.asarray(werte)
    bezug = np.asarray(bezug)
    return float(np.mean([funktion(werte[10 * i:10 * (i + 1)], bezug[10 * i:10 * (i + 1)]) for i in range(4)]))


def stichproben_fall(seed: int, familie: str, m: float, index: int, n: int = N_STICHPROBE, d: int = D) -> np.ndarray:
    """Stichprobe Y (n Punkte) aus N(μ, Σ) der Familie: Y = μ + Z Lᵀ mit der Cholesky-Zerlegung L von Σ; die Verschiebungsrichtung ist ein Strom des Zufalls-Seeds."""
    z = Z.stichproben(seed, ZWECK_Y, 1, n, d, start=index)[0]
    mu = np.zeros(d)
    if familie == "S":
        u = Z.zufall_normal(seed, ZWECK_U, 1, d, start=index)[0]
        mu = m * u / np.linalg.norm(u)
    return mu + z @ np.linalg.cholesky(sigma_matrix(familie, m, d)).T


def sliced_aus_stichproben(x: np.ndarray, y: np.ndarray, richtungen: np.ndarray) -> float:
    """Geschnittener W2 aus Stichproben: Wurzel des Mittels der quadrierten eindimensionalen Abstände über alle Richtungen (Zeilen von `richtungen`, Länge 1)."""
    px = np.sort(x @ richtungen.T, axis=0)
    py = np.sort(y @ richtungen.T, axis=0)
    return math.sqrt(float(np.mean((px - py) ** 2)))


def tabelle(seed: int, wdh: int = 20, n: int = N_STICHPROBE, verfahren: str = "auto") -> dict:
    """Die Zahlen des Tabs: Rangübereinstimmung von W2 aus Stichproben und SW2 (Population, aus Stichproben mit L Richtungen) mit der geschlossenen Wahrheit (Bures)."""
    fall = faelle()
    wahr = np.array([bures(f, m) for f, m in fall])
    sl_pop = np.array([sliced_population(f, m) for f, m in fall])
    x = Z.stichproben(seed, ZWECK_REF, 1, n, D)[0]
    ys = [stichproben_fall(seed, f, m, i, n) for i, (f, m) in enumerate(fall)]
    voll = np.array([math.sqrt(W.w2sq_zwei(x, y, verfahren)) for y in ys])
    rausch = float(np.mean([math.sqrt(W.w2sq_zwei(Z.stichproben(seed, ZWECK_REF, 1, n, D, start=10 + 2 * r)[0], Z.stichproben(seed, ZWECK_REF, 1, n, D, start=11 + 2 * r)[0], verfahren)) for r in range(5)]))
    zeilen = []
    for L in L_WERTE:
        sp, kt, spf = [], [], []
        for r in range(wdh):
            g = Z.zufall_normal(seed, ZWECK_RICHTUNG, 1, L * D, start=1000 * L + r)[0].reshape(L, D)
            g = g / np.linalg.norm(g, axis=1, keepdims=True)
            sl = np.array([sliced_aus_stichproben(x, y, g) for y in ys])
            sp.append(spearman(sl, wahr))
            kt.append(kendall(sl, wahr))
            spf.append(mittel_je_familie(sl, spearman, wahr))
        zeilen.append({"L": L, "spearman": float(np.mean(sp)), "kendall": float(np.mean(kt)), "spearman_familie": float(np.mean(spf))})
    return {
        "seed": seed, "wdh": wdh, "n": n, "rauschboden_w2": rausch, "bures_min": float(wahr.min()), "bures_max": float(wahr.max()),
        "voll_spearman": spearman(voll, wahr), "voll_kendall": kendall(voll, wahr), "voll_spearman_familie": mittel_je_familie(voll, spearman, wahr),
        "pop_spearman": spearman(sl_pop, wahr), "pop_kendall": kendall(sl_pop, wahr), "pop_spearman_familie": mittel_je_familie(sl_pop, spearman, wahr),
        "zeilen": zeilen,
    }
