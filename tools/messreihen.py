"""Rechnet die vorgerechneten Reihen der Demo und schreibt `tools/messreihen.json` (siehe dbd_vorgerechnet.py).

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/messreihen.py [Teil ...]
Teile: ref (Experiment 1: W1 durch Maßstab je d und n, Zufalls-Seed 2026), struktur (Experiment 3, 100 Wiederholungen), extrapolation (n = 2048 und 4096),
minibatch (Experiment 2, 2 000 Zufalls-Seeds und fünf Sätze für d = 2), wahrheit (d = 1, Zwei Stichproben gegen Wahrheit), streuung (Streuung der Steigung über Sätze von je 30 Wiederholungen), sliced (Tab). Ohne Teil: alle.
`OMP_NUM_THREADS=1` setzen; Dauer insgesamt einige Minuten, jeder Teil einzeln unter 8 Minuten. Bereits geschriebene Teile bleiben bei einem Teillauf erhalten."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import dbd_sliced as SL  # noqa: E402
import dbd_sweeps as S  # noqa: E402
from dbd_vorgerechnet import DATEI, quellen_pruefsumme  # noqa: E402

ZUFALLS_SEED = 2026
D_REIHE = (1, 2, 3, 4, 5, 10)
WDH_REF = {1: 200, 2: 30, 3: 30, 4: 30, 5: 30, 10: 30}
WDH_EXTRA = {1: 200, 2: 10, 3: 10, 4: 10, 5: 10, 10: 10}
STRUKTUREN = (("gerade", 0.0), ("gerade", 0.1), ("gerade", 0.3), ("gerade", 1.0), ("bloecke", 0.0), ("bloecke", 0.1), ("bloecke", 0.3), ("bloecke", 1.0))
WDH_STRUKTUR = 100
MINIBATCH_SEEDS = 2000
MINIBATCH_SAETZE = (2026, 2027, 2028, 2029, 2030)


def _liste(a):
    return [float(x) for x in a]


def teil_ref():
    ausgabe = {}
    for d in D_REIHE:
        r = S.reihe_normal(d, 1024, WDH_REF[d], ZUFALLS_SEED)
        ausgabe[str(d)] = {"wdh": WDH_REF[d], "ns": list(r["ns"]), "mittel": _liste(r["mittel"]), "verhaeltnis": _liste(r["verhaeltnis"]),
                           "se_verhaeltnis": _liste(r["w1"].std(axis=1, ddof=1) / np.sqrt(WDH_REF[d]) / r["massstab"]),
                           "steigung_alle": r["steigung_alle"], "steigung_letzte3": r["steigung_letzte3"]}
        print(f"  ref d = {d}: Steigung {r['steigung_alle']:+.3f} / {r['steigung_letzte3']:+.3f}", flush=True)
    return ausgabe


def teil_struktur():
    ausgabe = {}
    for familie, sigma in STRUKTUREN:
        r = S.reihe(familie, S.STRUKTUR_D, sigma, S.NS_ALLE, WDH_STRUKTUR, ZUFALLS_SEED)
        eintrag = {"wdh": WDH_STRUKTUR, "ns": list(r["ns"])}
        for wdh in (30, WDH_STRUKTUR):
            m = r["w1"][:, :wdh].mean(axis=1)
            eintrag[f"mittel_{wdh}"] = _liste(m)
            eintrag[f"steigung_alle_{wdh}"] = float(np.polyfit(np.log(r["ns"]), np.log(m), 1)[0])
            eintrag[f"steigung_letzte3_{wdh}"] = float(np.polyfit(np.log(r["ns"][-3:]), np.log(m[-3:]), 1)[0])
        ausgabe[f"{familie}_{sigma}"] = eintrag
        print(f"  struktur {familie} sigma = {sigma}: Steigung 30 Wdh {eintrag['steigung_alle_30']:+.3f} / {eintrag['steigung_letzte3_30']:+.3f}, "
              f"100 Wdh {eintrag['steigung_alle_100']:+.3f} / {eintrag['steigung_letzte3_100']:+.3f}", flush=True)
    return ausgabe


def teil_extrapolation(ref):
    ausgabe = {}
    for d in D_REIHE:
        r = ref[str(d)]
        t = time.time()
        zeilen = S.extrapolation(d, r["ns"], r["verhaeltnis"], wdh=WDH_EXTRA[d], seed=ZUFALLS_SEED)
        ausgabe[str(d)] = {"wdh": WDH_EXTRA[d], "zeilen": zeilen}
        print(f"  extrapolation d = {d}: " + ", ".join(f"n = {z['n']}: {z['verhaeltnis']:.3f} (SE {z['se'] / z['gemessen']:.1%})" for z in zeilen) + f" [{time.time() - t:.0f} s]", flush=True)
    return ausgabe


def teil_minibatch():
    ausgabe = {"seeds": MINIBATCH_SEEDS, "zufalls_seed": ZUFALLS_SEED, "reihen": {}, "saetze_d2": {}}
    for d in (2, 10):
        for form in ("gleich", "verschoben"):
            r = S.minibatch_reihe(d, form, MINIBATCH_SEEDS, ZUFALLS_SEED)
            ausgabe["reihen"][f"{d}_{form}"] = r
            print(f"  minibatch d = {d} {form}: k = 128: {r['zeilen'][-1]['mittel']:.4f}", flush=True)
    for satz in MINIBATCH_SAETZE:
        r = S.minibatch_reihe(2, "verschoben", MINIBATCH_SEEDS, satz, ks=(32, 128))
        ausgabe["saetze_d2"][str(satz)] = {"v32": r["zeilen"][0]["v"], "v128": r["zeilen"][1]["v"], "steigung": S.steigung_v_minus_1(r["zeilen"]),
                                           "anteil32": r["zeilen"][0]["anteil_unter_1"], "anteil128": r["zeilen"][1]["anteil_unter_1"]}
        print(f"  minibatch d = 2, Zufalls-Seed {satz}: V = {ausgabe['saetze_d2'][str(satz)]['v32']:.3f} / {ausgabe['saetze_d2'][str(satz)]['v128']:.3f}, Steigung {ausgabe['saetze_d2'][str(satz)]['steigung']:+.2f}", flush=True)
    return ausgabe


def teil_streuung():
    """Streuung der Steigung über unabhängige Sätze von je 30 Wiederholungen (Zufalls-Seeds 3000 bis 3000 + Sätze − 1)."""
    ausgabe = {}
    for name, familie, d, sigma, saetze in (("normal_1", "normal", 1, 0.0, 20), ("gerade_10", "gerade", 10, 0.0, 10), ("normal_10", "normal", 10, 0.0, 10)):
        alle, letzte = [], []
        for i in range(saetze):
            r = S.reihe(familie, d, sigma, S.NS_ALLE, 30, 3000 + i)
            alle.append(r["steigung_alle"])
            letzte.append(r["steigung_letzte3"])
        ausgabe[name] = {"saetze": saetze, "wdh": 30, "alle": _liste(alle), "letzte3": _liste(letzte),
                         "alle_mittel": float(np.mean(alle)), "alle_std": float(np.std(alle, ddof=1)), "letzte3_mittel": float(np.mean(letzte)), "letzte3_std": float(np.std(letzte, ddof=1)),
                         "letzte3_min": float(min(letzte)), "letzte3_max": float(max(letzte))}
        a = ausgabe[name]
        print(f"  streuung {name}: {saetze} Sätze, alle n {a['alle_mittel']:+.3f} (Std {a['alle_std']:.3f}), letzte drei n {a['letzte3_mittel']:+.3f} (Std {a['letzte3_std']:.3f}, {a['letzte3_min']:+.3f} bis {a['letzte3_max']:+.3f})", flush=True)
    return ausgabe


def teil_wahrheit():
    zeilen = [S.zwei_gegen_wahrheit(n, 200, ZUFALLS_SEED) for n in S.NS_ALLE]
    print("  wahrheit: Verhältnis je n " + " ".join(f"{z['verhaeltnis']:.3f}" for z in zeilen), flush=True)
    return {"wdh": 200, "zeilen": zeilen}


def teil_sliced():
    r = SL.tabelle(ZUFALLS_SEED)
    print(f"  sliced: Spearman Population {r['pop_spearman']:.3f}, L = 500: {r['zeilen'][-1]['spearman']:.3f}", flush=True)
    return r


def main():
    teile = sys.argv[1:] or ["ref", "struktur", "extrapolation", "minibatch", "wahrheit", "streuung", "sliced"]
    try:
        daten = json.loads(DATEI.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        daten = {}
    for teil in teile:
        t = time.time()
        print(f"{teil} ...", flush=True)
        if teil == "ref":
            daten["ref"] = teil_ref()
        elif teil == "struktur":
            daten["struktur"] = teil_struktur()
        elif teil == "extrapolation":
            daten["extrapolation"] = teil_extrapolation(daten["ref"])
        elif teil == "minibatch":
            daten["minibatch"] = teil_minibatch()
        elif teil == "streuung":
            daten["streuung"] = teil_streuung()
        elif teil == "wahrheit":
            daten["wahrheit"] = teil_wahrheit()
        elif teil == "sliced":
            daten["sliced"] = teil_sliced()
        else:
            raise SystemExit(f"unbekannter Teil {teil!r}")
        print(f"{teil}: {time.time() - t:.0f} s", flush=True)
        daten["quellen_pruefsumme"] = quellen_pruefsumme()
        DATEI.write_text(json.dumps(daten, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"geschrieben: {DATEI} ({DATEI.stat().st_size} Byte)")


if __name__ == "__main__":
    main()
