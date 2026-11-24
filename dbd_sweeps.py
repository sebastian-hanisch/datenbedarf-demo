"""Reihen der Experimente: Zwei-Stichproben-Abstand gegen n, Strukturen, Minibatch-Zuordnung, Extrapolation. Ohne Streamlit.

Zähleinheit jeder Mittelung ist die Wiederholung (Zufalls-Seed): eine Wiederholung zieht zwei unabhängige Stichproben der Größe n und rechnet den Abstand der exakten Zuordnung.
Jede Stichprobe ist ein eigener Strom (Zufalls-Seed der Demo, Zweck aus Familie und n, Zeile aus Wiederholung und Stichprobe); der Strom hängt nicht an der Wiederholungszahl
(30 Wiederholungen enthalten die ersten 10) und nicht an der Größe des n-Rasters."""

import math

import numpy as np

import dbd_wasserstein as W
import dbd_zufall as Z

NS_ALLE = (32, 64, 128, 256, 512, 1024)
BLOCK_ABSTAND = 3.0          # die zwei Blöcke liegen bei −3 e₁ und +3 e₁ (Abstand 6)
STRUKTUR_D = 10

ZWECK_NORMAL, ZWECK_GERADE, ZWECK_BLOECKE, ZWECK_BLOCKWAHL = 1, 2, 3, 4
ZWECK_MINIBATCH, ZWECK_EXTRA = 5, 6


def _zweck(code: int, n: int) -> int:
    return code * 10_000 + n


def stichprobe_paar(familie: str, d: int, sigma: float, n: int, seed: int, rep: int) -> tuple[np.ndarray, np.ndarray]:
    """Zwei unabhängige Stichproben (n Punkte) einer Familie in der Wiederholung `rep`.
    "normal": N(0, I_d). "gerade": Achse e₁ standardnormal, die übrigen d − 1 Achsen mit Streuung σ (σ = 0: die Punkte liegen auf einer Geraden; σ = 1: volle Dimension).
    "bloecke": N(0, σ² I_d) um −3 e₁ oder +3 e₁ (je Punkt mit Wahrscheinlichkeit ½); σ = 0: zwei Atome."""
    if familie == "normal":
        x = Z.stichproben(seed, _zweck(ZWECK_NORMAL, n), 2, n, d, start=2 * rep)
        return x[0], x[1]
    z = Z.stichproben(seed, _zweck(ZWECK_GERADE if familie == "gerade" else ZWECK_BLOECKE, n), 2, n, d, start=2 * rep)
    if familie == "gerade":
        z[:, :, 1:] *= sigma
        return z[0], z[1]
    if familie == "bloecke":
        z = z * sigma
        wahl = Z.zufall_gleich(seed, _zweck(ZWECK_BLOCKWAHL, n), 2, n, start=2 * rep)
        z[:, :, 0] += np.where(wahl < 0.5, -BLOCK_ABSTAND, BLOCK_ABSTAND)
        return z[0], z[1]
    raise ValueError(f"unbekannte Familie {familie!r}")


def w1_wiederholungen(familie: str, d: int, sigma: float, n: int, wdh: int, seed: int, verfahren: str = "auto") -> np.ndarray:
    """Zwei-Stichproben-W1 in `wdh` Wiederholungen bei Stichprobengröße n."""
    werte = []
    for r in range(wdh):
        x, y = stichprobe_paar(familie, d, sigma, n, seed, r)
        werte.append(W.w1_zwei(x, y, verfahren=verfahren))
    return np.array(werte)


def reihe(familie: str, d: int, sigma: float, ns, wdh: int, seed: int, verfahren: str = "auto") -> dict:
    """W1 je n und Wiederholung. Rückgabe: ns, w1 (Form (len(ns), wdh)), mittel (je n), steigung_alle (über alle n), steigung_letzte3 (über die letzten drei n)."""
    ns = tuple(int(n) for n in ns)
    w1 = np.array([w1_wiederholungen(familie, d, sigma, n, wdh, seed, verfahren) for n in ns])
    mittel = w1.mean(axis=1)
    return {
        "ns": ns, "wdh": wdh, "w1": w1, "mittel": mittel,
        "steigung_alle": W.steigung(ns, mittel),
        "steigung_letzte3": W.steigung(ns[-3:], mittel[-3:]) if len(ns) >= 3 else float("nan"),
    }


def ns_bis(n_max: int) -> tuple[int, ...]:
    """Das n-Raster 32, 64, ... bis n_max (Verdopplungen)."""
    return tuple(n for n in NS_ALLE if n <= n_max)


def reihe_normal(d: int, n_max: int, wdh: int, seed: int, verfahren: str = "auto") -> dict:
    """Experiment 1: Zwei-Stichproben-W1 im Standardnormal N(0, I_d), dazu W1 durch den Maßstab (mittleren Paarabstand) je n."""
    r = reihe("normal", d, 0.0, ns_bis(n_max), wdh, seed, verfahren)
    massstab = W.mittlerer_paarabstand(d)
    r["d"], r["massstab"] = d, massstab
    r["verhaeltnis"] = r["mittel"] / massstab
    r["steigung_alle"] = W.steigung(r["ns"], r["verhaeltnis"])
    r["steigung_letzte3"] = W.steigung(r["ns"][-3:], r["verhaeltnis"][-3:]) if len(r["ns"]) >= 3 else float("nan")
    return r


def noetiges_n(ns, verhaeltnis, ziel: float) -> tuple[float, bool]:
    """n, bei dem die Potenzanpassung a · n^b des Verhältnisses (W1 durch Maßstab) den Wert `ziel` erreicht, und ob das über das gemessene n-Raster hinausreicht (Stern)."""
    a, b = W.potenzanpassung(ns, verhaeltnis)
    n = (ziel / a) ** (1.0 / b)
    return n, n > max(ns)


# --- Minibatch ---------------------------------------------------------------------------------------------------------------------------------

MB_MU, MB_S = 2.0, 1.5       # Q = N(2 e₁, 1,5² I_d), P = N(0, I_d)
MB_KS = (8, 16, 32, 64, 128)
_BLOCK = 250                 # Batches je Stück (Speicher)


def w2sq_minibatch_wahr(d: int, form: str) -> float:
    """W2² zwischen P und Q: 0 bei gleichen Verteilungen, sonst μ² + d (s − 1)² (Gauß mit gleicher Drehung)."""
    return 0.0 if form == "gleich" else MB_MU ** 2 + d * (MB_S - 1.0) ** 2


def minibatch_werte(d: int, k: int, seeds: int, seed: int, form: str, verfahren: str = "auto") -> np.ndarray:
    """Minibatch-Wert je Zufalls-Seed: mittlere quadrierte Entfernung der optimalen Zuordnung zweier Batches der Größe k (X aus P, Y aus Q); absolut, nicht durch W2² geteilt."""
    werte = np.empty(seeds)
    zweck = _zweck(ZWECK_MINIBATCH, 100 * d + k)
    for start in range(0, seeds, _BLOCK):
        m = min(_BLOCK, seeds - start)
        z = Z.stichproben(seed, zweck, 2 * m, k, d, start=2 * start)        # Zeile 2j = X, 2j + 1 = Y des Batches start + j
        for j in range(m):
            x = z[2 * j]
            y = z[2 * j + 1]
            if form == "verschoben":
                y = y * MB_S
                y[:, 0] += MB_MU
            c = W.kosten_quadrat(x, y)
            col = W.zuordnung(c, verfahren)
            werte[start + j] = c[np.arange(k), col].mean()
    return werte


def minibatch_reihe(d: int, form: str, seeds: int, seed: int, ks=MB_KS, verfahren: str = "auto") -> dict:
    """Experiment 2 (Minibatch): je k Mittel, Standardfehler, V = Mittel durch W2² (bei verschoben), Anteil der Batches mit V < 1, dazu die Zufallspaarung 2d bei gleichen Verteilungen."""
    wahr = w2sq_minibatch_wahr(d, form)
    zeilen = []
    for k in ks:
        w = minibatch_werte(d, k, seeds, seed, form, verfahren)
        mittel, se = float(w.mean()), float(w.std(ddof=1) / math.sqrt(len(w)))
        zeilen.append({
            "k": k, "mittel": mittel, "se": se,
            "v": mittel / wahr if wahr > 0 else float("nan"), "v_se": se / wahr if wahr > 0 else float("nan"),
            "anteil_unter_1": float(np.mean(w < wahr)) if wahr > 0 else float("nan"),
        })
    return {"d": d, "form": form, "seeds": seeds, "wahr": wahr, "zufallspaarung": 2.0 * d if form == "gleich" else float("nan"), "zeilen": zeilen}


def steigung_v_minus_1(zeilen: list[dict], k0: int = 32, k1: int = 128) -> float:
    """Steigung von log(V − 1) gegen log k zwischen zwei Zellen (nur bei verschobenen Verteilungen)."""
    v = {z["k"]: z["v"] for z in zeilen}
    return math.log((v[k1] - 1.0) / (v[k0] - 1.0)) / math.log(k1 / k0)


# --- Extrapolation um den Faktor 4 --------------------------------------------------------------------------------------------------------------

def extrapolation(d: int, anpassung_ns, anpassung_verhaeltnis, ns=(2048, 4096), wdh: int = 10, seed: int = 2026, verfahren: str = "auto") -> list[dict]:
    """Messung bei n = 2048 und 4096 gegen die Vorhersage der Potenzanpassung aus dem n-Raster 32 bis 1 024: Verhältnis gemessen zu vorhergesagt und Standardfehler."""
    a, b = W.potenzanpassung(anpassung_ns, anpassung_verhaeltnis)
    massstab = W.mittlerer_paarabstand(d)
    zeilen = []
    for n in ns:
        w = np.array([W.w1_zwei(*stichprobe_paar("normal", d, 0.0, n, seed, r), verfahren=verfahren) for r in range(wdh)]) / massstab
        vorher = a * n ** b
        zeilen.append({"n": n, "gemessen": float(w.mean()), "se": float(w.std(ddof=1) / math.sqrt(wdh)), "vorhergesagt": float(vorher), "verhaeltnis": float(w.mean() / vorher)})
    return zeilen


# --- Zwei Stichproben gegen Wahrheit (d = 1) -----------------------------------------------------------------------------------------------------

def zwei_gegen_wahrheit(n: int, wdh: int, seed: int) -> dict:
    """d = 1: je Wiederholung eine Stichprobe X und eine zweite Y aus N(0, 1); Zwei-Stichproben-W1 (sortiert) und Abstand der Stichprobe X zur Wahrheit (Quadratur). Verhältnis der Mittel."""
    x = Z.zufall_normal(seed, _zweck(ZWECK_NORMAL, n), 2 * wdh, n)
    xs, ys = np.sort(x[0::2], axis=1), np.sort(x[1::2], axis=1)           # dieselben Ströme wie bei den Reihen mit d = 1
    zwei = np.mean(np.abs(xs - ys), axis=1)
    wahr = W.w1_wahrheit(xs)
    return {"n": n, "wdh": wdh, "zwei": float(zwei.mean()), "wahrheit": float(wahr.mean()), "verhaeltnis": float(zwei.mean() / wahr.mean())}
