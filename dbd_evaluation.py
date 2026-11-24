"""Einstellungen, Formatierung, Auswertung und Verdikte der drei Experimente (reine Funktionen, ohne Streamlit).

Alles Gerechnete kommt aus `dbd_sweeps.py`, `dbd_wasserstein.py` und `dbd_sliced.py`; die Verdikte bilden ihre Zahlen mit f-Strings aus dem Ergebnis."""

import math
from dataclasses import dataclass, fields

import numpy as np

import dbd_constants as C
import dbd_sliced as SL
import dbd_sweeps as S
import dbd_vorgerechnet as VG
import dbd_wasserstein as W


@dataclass(frozen=True)
class Settings:
    seed_input: int = C.DEFAULTS["seed_input"]
    d1: int = C.DEFAULTS["d1"]
    nmax1: int = C.DEFAULTS["nmax1"]
    wdh1: int = C.DEFAULTS["wdh1"]
    modus2: str = C.DEFAULTS["modus2"]
    n2: int = C.DEFAULTS["n2"]
    d2: int = C.DEFAULTS["d2"]
    wdh2: int = C.DEFAULTS["wdh2"]
    dm: int = C.DEFAULTS["dm"]
    formm: str = C.DEFAULTS["formm"]
    seedsm: int = C.DEFAULTS["seedsm"]
    km: int = C.DEFAULTS["km"]
    art3: str = C.DEFAULTS["art3"]
    sigma3: float = C.DEFAULTS["sigma3"]
    wdh3: int = C.DEFAULTS["wdh3"]


def settings_aus_zustand(zustand) -> Settings:
    """Settings aus einem Mapping (Session State oder Preset); fehlende Schlüssel nehmen den Standard."""
    return Settings(**{f.name: zustand.get(f.name, getattr(Settings(), f.name)) for f in fields(Settings)})


# ---------------------------------------------------------------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------------------------------------------------------------

def zahl(x: float, stellen: int = 3) -> str:
    """Dezimalkomma, Minuszeichen „−“; −0 wird zu 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    text = f"{x:.{stellen}f}".replace(".", ",")
    return text[1:] if text.startswith("-0") and float(text.replace(",", ".")) == 0 else text.replace("-", "−")


def ganz(n) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(n):,}".replace(",", " ")


def prozent(x: float, stellen: int = 0) -> str:
    return zahl(100.0 * x, stellen) + " %"


def gross_n(n: float) -> str:
    """Anzahl n: bis 99 999 ausgeschrieben (gerundet auf drei gültige Stellen), darüber in Zehnerpotenz-Schreibweise (7,1e8)."""
    if n < 1e5:
        schritt = 10 ** max(0, int(math.floor(math.log10(max(n, 1)))) - 2)
        return ganz(int(round(n / schritt)) * schritt)
    text = f"{n:.1e}".replace(".", ",")
    mant, exp = text.split("e")
    return f"{mant}e{int(exp)}"


def exp_text(x: float, stellen: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise mit Dezimalkomma, z. B. 1,2e-12."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{stellen}e}".split("e")
    return f"{mant.replace('.', ',')}e{int(exp)}"


def steigung_text(s: float, stellen: int = 2) -> str:
    return zahl(s, stellen)


def tabelle_md(kopf, zeilen) -> str:
    """Markdown-Tabelle aus Kopfzeile und Zeilen (Listen von Texten)."""
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Steigung gegen Dimension, nötiges n
# ---------------------------------------------------------------------------------------------------------------------------------

def erwartung_text(d: int) -> str:
    """Erwartete Steigung der oberen Schranke (Fournier/Guillin, Satz 1, p = 1) für N(0, I_d)."""
    if d == 1:
        return "−1/2"
    if d == 2:
        return "−1/2 mit Log-Faktor"
    return f"−1/{d}"


def analyse1(s: Settings, verfahren: str = "auto") -> dict:
    """Live-Reihe für die gewählte Dimension, dazu nötiges n und die vorgerechnete Messung bei n = 2 048 und 4 096."""
    r = S.reihe_normal(s.d1, s.nmax1, s.wdh1, s.seed_input, verfahren)
    r["noetig"] = {ziel: S.noetiges_n(r["ns"], r["verhaeltnis"], ziel) for ziel in (0.10, 0.05)}
    extra = VG.teil("extrapolation")
    r["extrapolation"] = extra.get(str(s.d1)) if extra else None
    return r


def referenz_reihen() -> dict | None:
    """Vorgerechnete Bezugsreihen je Dimension (Zufalls-Seed 2026): d -> {ns, verhaeltnis, steigung_alle, ...}."""
    return VG.teil("ref")


def verdikt1(s: Settings, r: dict) -> tuple[str, str]:
    d, ns = r["d"], r["ns"]
    n_max = ns[-1]
    anteil = r["verhaeltnis"][-1]
    teile = [
        f"d = {d}: Steigung {steigung_text(r['steigung_alle'])} zwischen n = 32 und {ganz(n_max)}"
        + (f", über die letzten drei n {steigung_text(r['steigung_letzte3'])}" if len(ns) >= 3 else "")
        + f" ({r['wdh']} Wiederholungen, Zufalls-Seed {s.seed_input}; Erwartung der oberen Schranke: {erwartung_text(d)}). "
        f"Bei n = {ganz(n_max)} liegt der Abstand zweier Stichproben aus derselben Verteilung noch bei {prozent(anteil)} des mittleren Paarabstands (Maßstab {zahl(r['massstab'], 2)}), obwohl der wahre Abstand null ist."
    ]
    n10, st10 = r["noetig"][0.10]
    n5, st5 = r["noetig"][0.05]
    if st10 or st5:
        gr = max(math.log10(n / n_max) for n, st in ((n10, st10), (n5, st5)) if st)       # größte der gesternten Zahlen, nicht nur die erste
        teile.append(
            f"Für 10 % des Paarabstands wäre n ≈ {gross_n(n10)}{'*' if st10 else ''}, für 5 % n ≈ {gross_n(n5)}{'*' if st5 else ''} "
            f"(* Extrapolation der Anpassung über n = {ganz(n_max)} hinaus, bis zu {zahl(gr, 1)} Größenordnungen, nicht gemessen)."
        )
    else:
        teile.append(f"Für 10 % des Paarabstands genügt n ≈ {gross_n(n10)}, für 5 % n ≈ {gross_n(n5)} (innerhalb des gemessenen Bereichs).")
    ex = r.get("extrapolation")
    if ex and n_max == 1024 and r["wdh"] == 30 and s.seed_input == C.DEFAULT_SEED:
        z = ex["zeilen"]
        teile.append(
            f"Um den Faktor 4 über das Raster hinaus (n = 2 048 und 4 096, {ex['wdh']} Wiederholungen) liegt die Messung bei {zahl(z[0]['verhaeltnis'], 2)} und {zahl(z[1]['verhaeltnis'], 2)} der Vorhersage der Anpassung."
        )
    if d == 1:
        teile.append("Die Steigung ist bei d = 1 etwa −0,5, nicht −1.")
    elif d >= 5:
        teile.append(f"Die gemessene Steigung ist flacher als −1/2 (bei d = {d}: {steigung_text(r['steigung_alle'])}); ein gemessener Abstand in der Höhe des Rauschbodens ({prozent(anteil)} des Paarabstands bei n = {ganz(n_max)}) ist ohne Vergleich mit ihm nicht als Abstand der Verteilungen zu lesen.")
    return ("warning" if d >= 5 else "info"), " ".join(teile)


def streuung_text() -> str:
    """Streuung der Steigung über die letzten drei n von Satz zu Satz (30 Wiederholungen, unabhängige Sätze), aus der vorgerechneten Messung."""
    st = VG.teil("streuung")
    if not st:
        return "Die Steigung über die letzten drei n streut von Satz zu Satz von Wiederholungen merklich (siehe README)."
    return (f"Die Steigung über die letzten drei n streut bei 30 Wiederholungen von Satz zu Satz mit einer Standardabweichung von {zahl(st['normal_1']['letzte3_std'], 2)} bei d = 1 und {zahl(st['gerade_10']['letzte3_std'], 2)} bei der Geraden in R^10 "
            f"(über alle n {zahl(st['normal_1']['alle_std'], 2)} und {zahl(st['gerade_10']['alle_std'], 2)}), bei d = 10 nur {zahl(st['normal_10']['letzte3_std'], 3)}.")


def halbierungsfaktor(steigung: float) -> float:
    """Faktor, um den n wachsen muss, damit der Abstand bei Potenzgesetz n^s auf die Hälfte fällt: 2^(1/|s|)."""
    return 2.0 ** (1.0 / abs(steigung))


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Rauschboden (Zwei Stichproben) und Minibatch
# ---------------------------------------------------------------------------------------------------------------------------------

def analyse2_zwei(s: Settings, verfahren: str = "auto") -> dict:
    """Zwei-Stichproben-W1 bei (n2, d2); bei d = 1 dazu der Abstand zur Wahrheit; die d = 1-Tabelle über das n-Raster; Zwei-Punkt-Fall exakt."""
    w1 = S.w1_wiederholungen("normal", s.d2, 0.0, s.n2, s.wdh2, s.seed_input, verfahren)
    massstab = W.mittlerer_paarabstand(s.d2)
    wahr = None
    if s.d2 == 1:
        wahr = S.zwei_gegen_wahrheit(s.n2, s.wdh2, s.seed_input)
    tab = [S.zwei_gegen_wahrheit(n, s.wdh2, s.seed_input) for n in S.NS_ALLE]
    return {
        "n": s.n2, "d": s.d2, "wdh": s.wdh2, "w1_mittel": float(w1.mean()), "w1_se": float(w1.std(ddof=1) / math.sqrt(len(w1))),
        "massstab": massstab, "anteil": float(w1.mean() / massstab), "wahrheit": wahr, "tabelle_d1": tab,
        "zwei_punkt": [{"N": N, "zwei": float(W.zwei_punkt_zwei_stichproben(N)), "wahr": float(W.zwei_punkt_wahrheit(N)), "verhaeltnis": W.zwei_punkt_verhaeltnis(N)} for N in (2, 4, 32, 1024)],
    }


def verdikt2_zwei(s: Settings, a: dict) -> tuple[str, str]:
    v = [z["verhaeltnis"] for z in a["tabelle_d1"]]
    teile = [
        f"d = 1: der Zwei-Stichproben-Abstand liegt im Mittel beim {zahl(float(np.mean(v)), 2)}-Fachen des Abstands zur Wahrheit (Mittel über n = 32 bis 1 024; "
        f"je n zwischen {zahl(min(v), 2)} und {zahl(max(v), 2)}; {a['wdh']} Wiederholungen, Standardnormal; mit weniger Wiederholungen streut das Verhältnis stärker). "
        f"Der Grenzwert ist √2 ≈ {zahl(math.sqrt(2), 2)}; im Zwei-Punkt-Fall ist das Verhältnis exakt {zahl(a['zwei_punkt'][0]['verhaeltnis'], 1)} bei N = 2 und fällt auf {zahl(a['zwei_punkt'][-1]['verhaeltnis'], 3)} bei N = 1 024."
    ]
    if s.d2 >= 2:
        teile.append(f"Für d ≥ 2 ist dieser Faktor nicht gemessen (die Wahrheit ist dort nicht geschlossen berechenbar); bei d = {s.d2} und n = {ganz(s.n2)} liegt der Abstand zweier Stichproben aus derselben Verteilung bei {prozent(a['anteil'])} des mittleren Paarabstands.")
    return "info", " ".join(teile)


def analyse2_mb(s: Settings, verfahren: str = "auto") -> dict:
    """Minibatch-Reihe über k für (dm, formm), dazu bei P = Q der Zwei-Stichproben-W2² mit n = k aus anderen Strömen (dieselbe Größe)."""
    form = "gleich" if s.formm == C.FORMM[1] else "verschoben"
    r = S.minibatch_reihe(s.dm, form, s.seedsm, s.seed_input, verfahren=verfahren)
    if form == "gleich":
        ref = []
        for k in S.MB_KS:
            w = np.array([W.w2sq_zwei(*S.stichprobe_paar("normal", s.dm, 0.0, k, s.seed_input, rep), verfahren=verfahren) for rep in range(min(s.seedsm, 500))])
            ref.append({"k": k, "mittel": float(w.mean()), "se": float(w.std(ddof=1) / math.sqrt(len(w))), "wdh": len(w)})
        r["zwei_stichproben"] = ref
    return r


def zeile_k(r: dict, k: int) -> dict:
    return next(z for z in r["zeilen"] if z["k"] == k)


def verdikt2_mb(s: Settings, r: dict) -> tuple[str, str]:
    d, k, seeds = r["d"], s.km, r["seeds"]
    z = zeile_k(r, k)
    if r["form"] == "gleich":
        text = (
            f"d = {d}, k = {k}: der Minibatch-Wert zwischen gleichen Verteilungen ist {zahl(z['mittel'], 2)} ± {zahl(z['se'], 2)} statt 0 ({ganz(seeds)} Zufalls-Seeds; die Zufallspaarung kostet {zahl(r['zufallspaarung'], 0)}). "
            f"Das ist derselbe Rauschboden wie bei zwei Stichproben mit n = k: der Zwei-Stichproben-W2² mit n = {k} ist {zahl(next(x for x in r['zwei_stichproben'] if x['k'] == k)['mittel'], 2)} ± {zahl(next(x for x in r['zwei_stichproben'] if x['k'] == k)['se'], 2)} "
            f"(andere Zufallsströme, {next(x for x in r['zwei_stichproben'] if x['k'] == k)['wdh']} Wiederholungen). Eine isotrope Gauß-Familie, exakte Zuordnung."
        )
        return "info", text
    unter = z["anteil_unter_1"]
    text = (
        f"d = {d}, k = {k}: V = Minibatch-Wert durch wahres W2² ({zahl(r['wahr'], 1)}) = {zahl(z['v'], 3)} ± {zahl(z['v_se'], 3)} ({ganz(seeds)} Zufalls-Seeds), im Mittel über dem wahren Wert. "
        f"{prozent(unter)} der einzelnen Batches liegen unter dem wahren W2², im Mittel liegt V nie darunter (gemeinsame Konvexität, eine Rechnung; im Zwei-Punkt-Beispiel exakt und in allen gemessenen Zellen so). Eine isotrope Gauß-Familie, exakte Zuordnung; andere Formen nicht gemessen."
    )
    if d == 2:
        text += " Bei d = 2 streut V − 1 von einem Satz Zufalls-Seeds zum anderen so stark, dass die Steigung von log(V − 1) nur als Band angegeben werden kann (siehe README)."
    return ("warning" if d == 10 else "info"), text


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Struktur
# ---------------------------------------------------------------------------------------------------------------------------------

def familie_sigma(s: Settings) -> tuple[str, float]:
    """(Familie, σ) der gewählten Datenart."""
    return {
        C.ART3[0]: ("gerade", 0.0),
        C.ART3[1]: ("gerade", s.sigma3),
        C.ART3[2]: ("bloecke", 0.0),
        C.ART3[3]: ("bloecke", s.sigma3),
    }[s.art3]


def exakte_steigung_atome(ns=S.NS_ALLE, abstand: float = 2 * S.BLOCK_ABSTAND) -> float:
    """Zwei Atome im Abstand Δ: E W1 = Δ · C(2n, n)/4^n; Steigung der Potenzanpassung über das n-Raster (exakt, ohne Zufall)."""
    return W.steigung(ns, [abstand * float(W.zwei_punkt_zwei_stichproben(n)) for n in ns])


def analyse3(s: Settings, verfahren: str = "auto") -> dict:
    familie, sigma = familie_sigma(s)
    r = S.reihe(familie, C.D3, sigma, S.NS_ALLE, s.wdh3, s.seed_input, verfahren)
    r["familie"], r["sigma"] = familie, sigma
    ref = referenz_reihen()
    r["volle_dimension"] = ref["10"] if ref else None
    if familie == "bloecke" and sigma == 0.0:
        r["exakt"] = exakte_steigung_atome()
    if familie == "gerade" and sigma == 0.0:
        x, y = S.stichprobe_paar("gerade", C.D3, 0.0, 128, s.seed_input, 0)
        r["gerade_gegen_1d"] = abs(W.w1_zwei(x, y) - W.w1_zwei(x[:, :1], y[:, :1]))
    return r


def verdikt3(s: Settings, r: dict) -> tuple[str, str]:
    voll = r["volle_dimension"]
    kopf = f"R^{C.D3}, {r['wdh']} Wiederholungen (Zufalls-Seed {s.seed_input}): {s.art3}"
    if r["familie"] == "gerade" and r["sigma"] == 0.0:
        wesen = f"Steigung {steigung_text(r['steigung_alle'])} (alle n), {steigung_text(r['steigung_letzte3'])} (letzte drei n): Die Punkte liegen auf einer Geraden, das ist ein eindimensionales Problem und verhält sich wie d = 1 (um −0,5; einzelne Werte weichen im Rahmen der unten genannten Streuung ab)."
    elif r["familie"] == "gerade":
        wesen = f"mit Streuung σ = {zahl(r['sigma'], 1)} auf den neun Nebenachsen: Steigung {steigung_text(r['steigung_alle'])} (alle n), {steigung_text(r['steigung_letzte3'])} (letzte drei n)."
    elif r["sigma"] == 0.0:
        wesen = (f"zwei Atome im Abstand 6: Steigung {steigung_text(r['steigung_alle'])} (alle n) gemessen, exakt {steigung_text(r['exakt'], 3)} aus der Binomialrechnung; die Dimension geht nicht ein.")
    else:
        wesen = f"zwei Blöcke mit Streuung σ = {zahl(r['sigma'], 1)} je Achse: Steigung {steigung_text(r['steigung_alle'])} (alle n), {steigung_text(r['steigung_letzte3'])} (letzte drei n)."
    voll_text = f" Zum Vergleich volle Dimension (Standardnormal in R^10, {voll['wdh']} Wiederholungen, Zufalls-Seed {C.DEFAULT_SEED}): {steigung_text(voll['steigung_alle'])}." if voll else ""
    ende = f" {streuung_text()} Die Untergrenze n^(−1/2) bei getrenntem Träger ist keine bessere Rate."
    return "info", f"{kopf}: {wesen}{voll_text}{ende}"


# ---------------------------------------------------------------------------------------------------------------------------------
# Tab sliced
# ---------------------------------------------------------------------------------------------------------------------------------

def tabelle_sliced() -> dict | None:
    return VG.teil("sliced")
