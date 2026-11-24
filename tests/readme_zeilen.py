"""Tabellen und Sätze der README, gebildet aus `tools/messreihen.json` und dem Code der Demo (dieselben Formatfunktionen wie in der App).

`tools/readme_bauen.py` setzt sie in `tools/readme_vorlage.md` ein; `tests/test_claims.py` prüft, dass jedes Stück wörtlich in der README steht. Keine Zahl der Befunde steht von Hand
in der README: ändert sich eine Messung, ändert sich das Stück, und der Test meldet die veraltete README."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import dbd_constants as C  # noqa: E402
import dbd_evaluation as E  # noqa: E402
import dbd_sliced as SL  # noqa: E402
import dbd_sweeps as S  # noqa: E402
import dbd_wasserstein as W  # noqa: E402

MESS = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))
REF, STRUKTUR, EXTRA, WAHR, MB, SLICED, STREU = MESS["ref"], MESS["struktur"], MESS["extrapolation"], MESS["wahrheit"], MESS["minibatch"], MESS["sliced"], MESS["streuung"]
Z, G, P = E.zahl, E.ganz, E.prozent


def tabelle_steigungen() -> str:
    zeilen = []
    for d in sorted(REF, key=int):
        r = REF[d]
        n10, st10 = S.noetiges_n(r["ns"], r["verhaeltnis"], 0.10)
        n5, st5 = S.noetiges_n(r["ns"], r["verhaeltnis"], 0.05)
        zeilen.append([d, str(r["wdh"]), Z(r["steigung_alle"], 3), Z(r["steigung_letzte3"], 3), P(r["verhaeltnis"][0], 1), P(r["verhaeltnis"][-1], 1),
                       E.gross_n(n10) + ("*" if st10 else ""), E.gross_n(n5) + ("*" if st5 else ""), Z(E.halbierungsfaktor(r["steigung_alle"]), 0)])
    return E.tabelle_md(["d", "Wiederholungen", "Steigung (alle n)", "Steigung (letzte drei n)", "W1 durch Maßstab, n = 32", "n = 1 024", "n für 10 %", "n für 5 %", "Halbierungsfaktor"], zeilen)


def satz_rauschboden() -> str:
    return (f"Bei n = 1 024 liegt der Zwei-Stichproben-Abstand bei d = 10 bei {P(REF['10']['verhaeltnis'][-1])} des mittleren Paarabstands, bei d = 5 bei {P(REF['5']['verhaeltnis'][-1])} und bei d = 1 bei {P(REF['1']['verhaeltnis'][-1])}; "
            f"der wahre Abstand ist in allen Fällen null.")


def satz_halbierung() -> str:
    return (f"Aus der gemessenen Steigung folgt (Rechnung, vorasymptotisch) ein Halbierungsfaktor 2^(1/|Steigung|) von {Z(E.halbierungsfaktor(REF['5']['steigung_alle']), 0)} bei d = 5 und {Z(E.halbierungsfaktor(REF['10']['steigung_alle']), 0)} bei d = 10; "
            f"die Schranke mit Steigung −1/d gäbe {2 ** 5} und {2 ** 10}.")


MC_ERWARTUNG_D1_N32 = 0.2676       # Erwartung von W1 durch Maßstab bei d = 1, n = 32 (Monte Carlo, 100 000 weitere Wiederholungen); `test_claims.py` rechnet sie als Band neu


def satz_zellen() -> str:
    """Streuung der Tabellenzellen (Standardfehler der Mittel) und der Stern bei d = 1."""
    ds = sorted(REF, key=int)
    se32 = " / ".join(Z(100 * REF[d]["se_verhaeltnis"][0], 1) for d in ds)
    se1024 = " / ".join(Z(100 * REF[d]["se_verhaeltnis"][-1], 2) for d in ds)
    r1 = REF["1"]
    z32, se = r1["verhaeltnis"][0], r1["se_verhaeltnis"][0]
    a, b = W.potenzanpassung(r1["ns"], r1["verhaeltnis"])
    n5, _ = S.noetiges_n(r1["ns"], r1["verhaeltnis"], 0.05)
    return (f"Jede Zelle ist ein Mittel über {r1['wdh']} (d = 1) beziehungsweise 30 Wiederholungen und streut: Standardfehler der Zellen in Prozentpunkten bei n = 32 {se32} und bei n = 1 024 {se1024} "
            f"(d = {' / '.join(ds)}). Beispiel d = 1, n = 32: die Zelle ({P(z32, 1)}, Standardfehler {Z(100 * se, 1)}) liegt rund {Z((z32 - MC_ERWARTUNG_D1_N32) / se, 1)} Standardfehler über der Erwartung {P(MC_ERWARTUNG_D1_N32, 1)} "
            f"(Mittel über 100 000 weitere Wiederholungen mit einem anderen Zufallsgenerator), also am oberen Rand dessen, was beim Zufalls-Seed 2026 vorkommt; die Steigung bei d = 1 ist davon kaum betroffen. "
            f"Der Stern bei d = 1 (n für 5 % = {E.gross_n(n5)}*) gilt für die Anpassung: sie erreicht 5 % erst bei n = {E.gross_n(n5)}, {P(a * 1024.0 ** b, 2)} bei n = 1 024, während der gemessene Mittelwert bei n = 1 024 mit {P(r1['verhaeltnis'][-1], 1)} "
            f"schon unter 5 % liegt (Standardfehler {Z(100 * r1['se_verhaeltnis'][-1], 2)} Prozentpunkte); der Abstand zum Raster ist kleiner als diese Streuung.")


def tabelle_extrapolation() -> str:
    zeilen = []
    for d in sorted(EXTRA, key=int):
        z0, z1 = EXTRA[d]["zeilen"]
        zeilen.append([d, str(EXTRA[d]["wdh"]), f"{Z(z0['verhaeltnis'], 3)} (± {P(z0['se'] / z0['gemessen'], 1)})", f"{Z(z1['verhaeltnis'], 3)} (± {P(z1['se'] / z1['gemessen'], 1)})"])
    return E.tabelle_md(["d", "Wiederholungen", "n = 2 048", "n = 4 096"], zeilen)


def satz_extrapolation() -> str:
    abw = [(abs(z["verhaeltnis"] - 1.0), d, z["n"]) for d, v in EXTRA.items() for z in v["zeilen"]]
    groesste = max(abw)
    d_max, n_max = groesste[1], groesste[2]
    return (f"Die größte Abweichung der Messung von der Vorhersage beträgt {P(groesste[0], 1)} (d = {d_max}, n = {G(n_max)}); in allen zwölf Zellen liegt die Messung zwischen {Z(min(z['verhaeltnis'] for v in EXTRA.values() for z in v['zeilen']), 3)} und "
            f"{Z(max(z['verhaeltnis'] for v in EXTRA.values() for z in v['zeilen']), 3)} der Vorhersage, in {sum(1 for v in EXTRA.values() for z in v['zeilen'] if z['verhaeltnis'] < 1.0)} von 12 darunter.")


def tabelle_wahrheit() -> str:
    zeilen = [[G(z["n"]), Z(z["zwei"], 4), Z(z["wahrheit"], 4), Z(z["verhaeltnis"], 2)] for z in WAHR["zeilen"]]
    return E.tabelle_md(["n", "Zwei Stichproben (W1)", "Stichprobe gegen Wahrheit (W1)", "Verhältnis"], zeilen)


def satz_wahrheit() -> str:
    v = [z["verhaeltnis"] for z in WAHR["zeilen"]]
    return (f"Mittel der sechs Verhältnisse {Z(float(np.mean(v)), 2)}, je n zwischen {Z(min(v), 2)} und {Z(max(v), 2)} ({WAHR['wdh']} Wiederholungen je n, d = 1, Standardnormal); der Grenzwert ist √2 = {Z(math.sqrt(2), 3)}.")


def satz_zwei_punkt() -> str:
    return (f"Im Zwei-Punkt-Fall ist das Verhältnis exakt {Z(W.zwei_punkt_verhaeltnis(2), 1)} bei N = 2, {Z(W.zwei_punkt_verhaeltnis(4), 4)} bei N = 4 (35/24), {Z(W.zwei_punkt_verhaeltnis(32), 4)} bei N = 32 und {Z(W.zwei_punkt_verhaeltnis(1024), 4)} bei N = 1 024, "
            f"fallend gegen √2.")


def tabelle_minibatch(d: int) -> str:
    r = MB["reihen"][f"{d}_verschoben"]
    g = MB["reihen"][f"{d}_gleich"]
    zeilen = [[str(z["k"]), f"{Z(z['v'], 3)} ± {Z(z['v_se'], 3)}", P(z["anteil_unter_1"], 1), f"{Z(gz['mittel'], 3)} ± {Z(gz['se'], 3)}"] for z, gz in zip(r["zeilen"], g["zeilen"])]
    return E.tabelle_md(["k", f"V bei d = {d} (verschoben und gestreckt)", "Batches mit V unter 1", f"Wert bei P = Q, d = {d} (wahrer Abstand 0; Zufallspaarung {2 * d})"], zeilen)


def satz_minibatch_d10() -> str:
    r, g = MB["reihen"]["10_verschoben"]["zeilen"], MB["reihen"]["10_gleich"]["zeilen"]
    return (f"Bei d = 10 und k = 128 ist V = {Z(r[-1]['v'], 3)} ± {Z(r[-1]['v_se'], 3)} und der Wert zwischen gleichen Verteilungen {Z(g[-1]['mittel'], 2)} ± {Z(g[-1]['se'], 2)} statt 0 "
            f"({G(MB['seeds'])} Zufalls-Seeds, Zufalls-Seed {MB['zufalls_seed']}); keiner der {G(MB['seeds'])} Batches liegt bei d = 10 unter dem wahren W2² (Anteil {P(max(z['anteil_unter_1'] for z in r), 1)}).")


def tabelle_saetze_d2() -> str:
    zeilen = []
    for satz, v in MB["saetze_d2"].items():
        zeilen.append([satz, Z(v["v32"], 3), Z(v["v128"], 3), Z(v["steigung"], 2), P(v["anteil32"], 1), P(v["anteil128"], 1)])
    return E.tabelle_md(["Zufalls-Seed (Satz von 2 000)", "V bei k = 32", "V bei k = 128", "Steigung von log(V − 1), k = 32 bis 128", "Batches V < 1, k = 32", "k = 128"], zeilen)


def satz_minibatch_d2_band() -> str:
    s = MB["saetze_d2"].values()
    return (f"Über die fünf Sätze von je {G(MB['seeds'])} Zufalls-Seeds liegt V bei d = 2 und k = 128 zwischen {Z(min(v['v128'] for v in s), 3)} und {Z(max(v['v128'] for v in s), 3)}, bei k = 32 zwischen {Z(min(v['v32'] for v in s), 3)} und {Z(max(v['v32'] for v in s), 3)}; "
            f"die Steigung von log(V − 1) zwischen {Z(min(v['steigung'] for v in s), 2)} und {Z(max(v['steigung'] for v in s), 2)}; der Anteil der Batches unter dem wahren W2² zwischen {P(min(min(v['anteil32'], v['anteil128']) for v in s), 1)} und {P(max(max(v['anteil32'], v['anteil128']) for v in s), 1)}.")


def tabelle_struktur() -> str:
    namen = {"gerade_0.0": "Gerade (σ = 0)", "gerade_0.1": "Gerade + Rauschen σ = 0,1", "gerade_0.3": "Gerade + Rauschen σ = 0,3", "gerade_1.0": "Gerade + Rauschen σ = 1 (volle Dimension)",
             "bloecke_0.0": "Zwei Atome (σ = 0)", "bloecke_0.1": "Zwei Blöcke σ = 0,1", "bloecke_0.3": "Zwei Blöcke σ = 0,3", "bloecke_1.0": "Zwei Blöcke σ = 1"}
    zeilen = []
    for k, name in namen.items():
        v = STRUKTUR[k]
        zeilen.append([name, f"{Z(v['steigung_alle_30'], 2)} / {Z(v['steigung_letzte3_30'], 2)}", f"{Z(v['steigung_alle_100'], 2)} / {Z(v['steigung_letzte3_100'], 2)}", f"{Z(v['mittel_100'][0], 3)} / {Z(v['mittel_100'][-1], 3)}"])
    return E.tabelle_md(["Daten in R^10", "Steigung alle n / letzte drei n, 30 Wiederholungen", "100 Wiederholungen", "W1 bei n = 32 / 1 024 (100 Wiederholungen)"], zeilen)


def satz_atome_exakt() -> str:
    return (f"Zwei Atome im Abstand 6: exakt E W1 = 6 · C(2n, n)/4^n = {Z(6 * float(W.zwei_punkt_zwei_stichproben(32)), 4)} bei n = 32 und {Z(6 * float(W.zwei_punkt_zwei_stichproben(1024)), 4)} bei n = 1 024, Steigung {Z(E.exakte_steigung_atome(), 3)} "
            f"(Binomialrechnung, kein Zufall); die gemessene Steigung mit 100 Wiederholungen ist {Z(STRUKTUR['bloecke_0.0']['steigung_alle_100'], 3)}, mit 30 Wiederholungen {Z(STRUKTUR['bloecke_0.0']['steigung_alle_30'], 3)}: das Rauschen der Wiederholungen, nicht ein anderer Wert.")


def tabelle_streuung() -> str:
    namen = {"normal_1": "d = 1, Standardnormal", "gerade_10": "Gerade in R^10", "normal_10": "d = 10, Standardnormal"}
    zeilen = []
    for k, name in namen.items():
        v = STREU[k]
        zeilen.append([name, str(v["saetze"]), f"{Z(v['alle_mittel'], 3)} (Standardabweichung {Z(v['alle_std'], 3)})", f"{Z(v['letzte3_mittel'], 3)} (Standardabweichung {Z(v['letzte3_std'], 3)}; {Z(v['letzte3_min'], 2)} bis {Z(v['letzte3_max'], 2)})"])
    return E.tabelle_md(["Daten", "unabhängige Sätze von je 30 Wiederholungen", "Steigung alle n, Mittel über die Sätze", "Steigung letzte drei n, Mittel über die Sätze"], zeilen)


def satz_gerade() -> str:
    g = STRUKTUR["gerade_0.0"]
    return (f"Die Gerade in R^10 hat mit 30 Wiederholungen die Steigung {Z(g['steigung_alle_30'], 3)} (letzte drei n {Z(g['steigung_letzte3_30'], 3)}), mit 100 Wiederholungen {Z(g['steigung_alle_100'], 3)} ({Z(g['steigung_letzte3_100'], 3)}); "
            f"über {STREU['gerade_10']['saetze']} unabhängige Sätze von je 30 Wiederholungen liegt die Steigung bei {Z(STREU['gerade_10']['alle_mittel'], 3)} (alle n) und {Z(STREU['gerade_10']['letzte3_mittel'], 3)} (letzte drei n), "
            f"die Kontrolle d = 1 bei {Z(STREU['normal_1']['alle_mittel'], 3)} und {Z(STREU['normal_1']['letzte3_mittel'], 3)}: eine Gerade verhält sich wie d = 1.")


def satz_fazit_struktur() -> str:
    g = [STRUKTUR[f"gerade_{s}"]["steigung_alle_100"] for s in ("0.1", "0.3", "1.0")]
    b = [STRUKTUR[f"bloecke_{s}"]["steigung_alle_100"] for s in ("0.1", "0.3", "1.0")]
    return (f"Schon kleine Streuung (σ = 0,1 bis 0,3) senkt die Steigung (alle n, 100 Wiederholungen) bei der Geraden auf {Z(g[0], 2)} bis {Z(g[1], 2)}, bei zwei Blöcken auf {Z(b[0], 2)} bis {Z(b[1], 2)}; "
            f"bei σ = 1 (volle Dimension) liegt sie bei {Z(g[2], 2)} (Gerade) und {Z(b[2], 2)} (zwei Blöcke), bei d = 5 und d = 10 in der Reihe von Experiment 1 bei {Z(REF['5']['steigung_alle'], 2)} und {Z(REF['10']['steigung_alle'], 2)}. "
            f"Die Gerade erreicht diese Werte ab σ = 0,1 bis 0,3, zwei Blöcke mit σ = 0,1 ({Z(b[0], 2)}) noch nicht. Struktur hilft in dem Maß, in dem sie exakt ist (n = 32 bis 1 024, vorasymptotisch; nur vier Werte von σ gemessen).")


def tabelle_sliced_verhaeltnisse() -> str:
    ver = SL.verhaeltnisse()
    zeilen = [[f"{f} {SL.FAMILIEN_NAMEN[f]}", Z(float(np.mean(ver[f])), 3), Z(min(ver[f]), 3), Z(max(ver[f]), 3)] for f in SL.FAMILIEN]
    return E.tabelle_md(["Familie", "Mittel über zehn Stärken", "kleinster Wert", "größter Wert"], zeilen)


def tabelle_sliced_raenge() -> str:
    s = SLICED
    zeilen = [["voller Abstand W2 aus Stichproben (eine Stichprobe je Fall, ohne Streuungsangabe)",Z(s["voll_spearman"], 3), Z(s["voll_kendall"], 3), Z(s["voll_spearman_familie"], 3)],
              ["sliced, Population (geschlossen)", Z(s["pop_spearman"], 3), Z(s["pop_kendall"], 3), Z(s["pop_spearman_familie"], 3)]]
    zeilen += [[f"sliced aus Stichproben, L = {z['L']}", Z(z["spearman"], 3), Z(z["kendall"], 3), Z(z["spearman_familie"], 3)] for z in s["zeilen"]]
    return E.tabelle_md(["Abstand", "Spearman", "Kendall τ", "Spearman je Familie"], zeilen)


def satz_sliced_rauschboden() -> str:
    s = SLICED
    return f"Rauschboden des vollen W2 zwischen zwei Stichproben aus N(0, I_10) mit n = {s['n']}: {Z(s['rauschboden_w2'], 2)} (Mittel über 5 Paare); die Wahrheit der 40 Fälle liegt zwischen {Z(s['bures_min'], 2)} und {Z(s['bures_max'], 2)}."


def satz_streuung() -> str:
    return E.streuung_text()


def satz_jenseits() -> str:
    gr = [(math.log10(S.noetiges_n(REF[d]["ns"], REF[d]["verhaeltnis"], 0.05)[0] / 1024.0), d) for d in REF]
    return f"Die Werte mit Stern jenseits von n = 4 096 reichen bei d = 10 bis zu {Z(max(gr)[0], 1)} Größenordnungen über n = 1 024 hinaus (für 5 %); sie sind Rechenbeispiele, keine Messung."


def liste_korrekturen() -> str:
    v = [z["verhaeltnis"] for z in WAHR["zeilen"]]
    g = STRUKTUR["gerade_0.0"]
    s = list(MB["saetze_d2"].values())
    return "\n".join([
        "- **d = 1: Steigung etwa −0,5, nicht −1.** Die Formel „Rate n^(−1/d)“ steht nur mit „ab d = 3“; bei d = 1 und 2 gelten die Fälle nach Fournier und Guillin.",
        f"- **Faktor Zwei Stichproben gegen Wahrheit:** gemessen {Z(float(np.mean(v)), 2)} im Mittel (je n zwischen {Z(min(v), 2)} und {Z(max(v), 2)}), Grenzwert √2; nur bei d = 1 und nur für das Standardnormal.",
        f"- **Gerade:** die Steigung ist die von d = 1, aber mit 30 Wiederholungen streut sie über die letzten drei n von Satz zu Satz mit einer Standardabweichung von {Z(STREU['gerade_10']['letzte3_std'], 2)}; "
        f"ein einzelner Wert wie {Z(g['steigung_letzte3_30'], 2)} liegt rund {Z(abs(g['steigung_letzte3_30'] - STREU['gerade_10']['letzte3_mittel']) / STREU['gerade_10']['letzte3_std'], 1)} Standardabweichungen neben dem Mittel {Z(STREU['gerade_10']['letzte3_mittel'], 2)} der {STREU['gerade_10']['saetze']} Sätze, "
        f"ist also mit dem Rauschen vereinbar und kein Gegenbeispiel (100 Wiederholungen: {Z(g['steigung_letzte3_100'], 2)}).",
        f"- **Zwei Atome:** die Steigung ist exakt {Z(E.exakte_steigung_atome(), 3)}; eine Messreihe mit 100 Wiederholungen ergibt {Z(STRUKTUR['bloecke_0.0']['steigung_alle_100'], 2)} (Rauschen der Wiederholungen).",
        f"- **Minibatch bei d = 2:** V bei k = 128 liegt in fünf Sätzen von je {G(MB['seeds'])} Zufalls-Seeds zwischen {Z(min(x['v128'] for x in s), 3)} und {Z(max(x['v128'] for x in s), 3)}; bei k = 32 weichen die Sätze um bis zu {Z(max(x['v32'] for x in s) - min(x['v32'] for x in s), 3)} voneinander ab, "
        f"die Steigung von log(V − 1) um bis zu {Z(max(x['steigung'] for x in s) - min(x['steigung'] for x in s), 2)}. Die Zahl wird mit ihrer Zahl der Zufalls-Seeds genannt, nie allein.",
    ])


STUECKE = {
    "tabelle_steigungen": tabelle_steigungen,
    "satz_rauschboden": satz_rauschboden,
    "satz_halbierung": satz_halbierung,
    "satz_zellen": satz_zellen,
    "satz_fazit_struktur": satz_fazit_struktur,
    "tabelle_extrapolation": tabelle_extrapolation,
    "satz_extrapolation": satz_extrapolation,
    "tabelle_wahrheit": tabelle_wahrheit,
    "satz_wahrheit": satz_wahrheit,
    "satz_zwei_punkt": satz_zwei_punkt,
    "tabelle_minibatch_d2": lambda: tabelle_minibatch(2),
    "tabelle_minibatch_d10": lambda: tabelle_minibatch(10),
    "satz_minibatch_d10": satz_minibatch_d10,
    "tabelle_saetze_d2": tabelle_saetze_d2,
    "satz_minibatch_d2_band": satz_minibatch_d2_band,
    "tabelle_struktur": tabelle_struktur,
    "satz_atome_exakt": satz_atome_exakt,
    "tabelle_streuung": tabelle_streuung,
    "satz_streuung": satz_streuung,
    "satz_jenseits": satz_jenseits,
    "liste_korrekturen": liste_korrekturen,
    "satz_gerade": satz_gerade,
    "tabelle_sliced_verhaeltnisse": tabelle_sliced_verhaeltnisse,
    "tabelle_sliced_raenge": tabelle_sliced_raenge,
    "satz_sliced_rauschboden": satz_sliced_rauschboden,
}

if __name__ == "__main__":
    for name, f in STUECKE.items():
        print(f"--- {name}\n{f()}\n")
