"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a).

Dieser Test macht fünf Dinge:
(1) die Tabellen und Sätze der README werden aus `tools/messreihen.json` mit denselben Formatfunktionen wie in der App gebildet (`tests/readme_zeilen.py`) und müssen wörtlich in der README stehen;
(2) die Aussagen „in k von m Zellen“ und die Bänder der README werden an der Messdatei neu gezählt;
(3) Zellen der Messdatei werden mit dem Code der Demo neu gerechnet und als Band verglichen (die Normalzahlen sind in der letzten Binärstelle plattformabhängig); die Prüfsumme der Quellen
    sagt, ob die Datei zum Code gehört;
(4) die Zahlen, die die App fest im Text nennt (Intro, Grenzen, Kernfragen), stimmen mit der Messdatei überein;
(5) Konstanten, Kantenwortlaut, Testzahl und verbotene Wörter."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import dbd_constants as C
import dbd_evaluation as E
import dbd_sliced as SL
import dbd_sweeps as S
import dbd_vorgerechnet as VG
import dbd_wasserstein as W
import readme_zeilen as R

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
MESS = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))
REF, STRUKTUR, EXTRA, WAHR, MB, SLICED, STREU = MESS["ref"], MESS["struktur"], MESS["extrapolation"], MESS["wahrheit"], MESS["minibatch"], MESS["sliced"], MESS["streuung"]


# --- (1) Tabellen und Sätze wörtlich ---------------------------------------------------------------------------------------------------------

def test_tabellen_und_saetze_der_readme():
    for name, f in R.STUECKE.items():
        assert f() in README, name


def test_readme_hat_keine_offenen_platzhalter():
    assert "{{" not in README and "}}" not in README


# --- (2) Zählaussagen und Bänder -------------------------------------------------------------------------------------------------------------

def test_zaehlaussagen_der_readme():
    zellen = [z for v in EXTRA.values() for z in v["zeilen"]]
    unter = sum(1 for z in zellen if z["verhaeltnis"] < 1.0)
    assert len(zellen) == 12 and f"in {unter} von 12 darunter" in README
    abw = max(abs(z["verhaeltnis"] - 1.0) for z in zellen)
    assert f"beträgt {E.prozent(abw, 1)}" in README
    # kein Batch unter dem wahren W2² bei d = 10 (alle fünf k, 2 000 Zufalls-Seeds)
    assert all(z["anteil_unter_1"] == 0.0 for z in MB["reihen"]["10_verschoben"]["zeilen"]) and "(Anteil 0,0 %)" in README
    # bei d = 2 liegt der Anteil in allen fünf Sätzen und beiden k zwischen 24 und 37 Prozent
    anteile = [a for v in MB["saetze_d2"].values() for a in (v["anteil32"], v["anteil128"])]
    assert 0.24 < min(anteile) and max(anteile) < 0.37 and all(0.24 < z["anteil_unter_1"] < 0.37 for z in MB["reihen"]["2_verschoben"]["zeilen"])
    # V liegt im Mittel über 1 in jeder Zelle beider Dimensionen
    assert all(z["v"] > 1.0 for r in ("2_verschoben", "10_verschoben") for z in MB["reihen"][r]["zeilen"])
    # die Basislinie fällt mit k und liegt unter der Zufallspaarung (alle zehn Zellen)
    for d in (2, 10):
        m = [z["mittel"] for z in MB["reihen"][f"{d}_gleich"]["zeilen"]]
        assert all(a > b for a, b in zip(m, m[1:])) and max(m) < 2 * d


def test_steigungen_der_readme_liegen_in_den_genannten_baendern():
    # d = 1: etwa −0,5; d = 10: flach; Dimension senkt die Steigung monoton (alle sechs d)
    s = [REF[d]["steigung_alle"] for d in ("1", "2", "3", "4", "5", "10")]
    assert -0.52 < s[0] < -0.46 and -0.14 < s[-1] < -0.09 and all(a < b for a, b in zip(s, s[1:]))
    # Rauschboden 46 %, 27 %, 5 % bei n = 1 024
    assert [round(100 * REF[d]["verhaeltnis"][-1]) for d in ("10", "5", "1")] == [46, 27, 5]
    # Struktur: Steigung wächst mit der Streuung zur Dimension (alle n, 100 Wiederholungen), Gerade und Atome nahe −0,5
    g = [STRUKTUR[f"gerade_{s_}"]["steigung_alle_100"] for s_ in ("0.0", "0.1", "0.3", "1.0")]
    assert g[0] < -0.4 and all(a < b for a, b in zip(g, g[1:])) and abs(g[-1] - REF["10"]["steigung_alle"]) < 0.01
    assert abs(E.exakte_steigung_atome() + 0.499) < 0.0006
    # Streuung der Steigung: die Standardabweichung über die letzten drei n liegt bei d = 1 und der Geraden zwischen 0,05 und 0,09, bei d = 10 unter 0,005
    assert 0.05 < STREU["normal_1"]["letzte3_std"] < 0.09 and 0.05 < STREU["gerade_10"]["letzte3_std"] < 0.09 and STREU["normal_10"]["letzte3_std"] < 0.005
    # Faktor Zwei Stichproben gegen Wahrheit: je n zwischen 1,35 und 1,45
    v = [z["verhaeltnis"] for z in WAHR["zeilen"]]
    assert 1.35 <= min(v) and max(v) <= 1.45 and 1.38 < float(np.mean(v)) < 1.42


def test_sliced_aussagen_der_readme():
    ver = SL.verhaeltnisse()
    assert all(abs(x - 1 / math.sqrt(10)) < 1e-9 for f in "SI" for x in ver[f])
    assert abs(float(np.mean(ver["E"])) - 0.2097) < 5e-4 and abs(float(np.mean(ver["K"])) - 0.1553) < 5e-4
    assert SLICED["pop_spearman_familie"] > 0.9999 and 0.92 < SLICED["pop_spearman"] < 0.94
    sp = [z["spearman"] for z in SLICED["zeilen"]]
    assert all(a < b for a, b in zip(sp, sp[1:])) and sp[-1] < SLICED["pop_spearman"] + 0.02


# --- (3) Neu rechnen als Band, Prüfsumme -----------------------------------------------------------------------------------------------------

def test_quellen_pruefsumme_der_messreihen():
    assert MESS["quellen_pruefsumme"] == VG.quellen_pruefsumme(), "tools/messreihen.py neu laufen lassen (die Quellen haben sich geändert)"


def test_vorgerechnete_zellen_stimmen_beim_neu_rechnen():
    # Bezugsreihe: d = 2 und d = 10 bei n = 32 und 64, 30 Wiederholungen
    for d in (2, 10):
        r = S.reihe_normal(d, 64, REF[str(d)]["wdh"], 2026)
        assert np.allclose(r["verhaeltnis"], REF[str(d)]["verhaeltnis"][:2], rtol=1e-6), d
    # d = 1 mit 200 Wiederholungen bis n = 1 024 und die Steigung
    r1 = S.reihe_normal(1, 1024, 200, 2026)
    assert np.allclose(r1["verhaeltnis"], REF["1"]["verhaeltnis"], rtol=1e-6) and r1["steigung_alle"] == pytest.approx(REF["1"]["steigung_alle"], abs=1e-6)
    # Struktur: Gerade mit Rauschen 0,1 und zwei Blöcke 0,3, n = 32 und 64, 30 Wiederholungen
    for fam, sig, key in (("gerade", 0.1, "gerade_0.1"), ("bloecke", 0.3, "bloecke_0.3")):
        r = S.reihe(fam, 10, sig, (32, 64), 30, 2026)
        assert np.allclose(r["mittel"], STRUKTUR[key]["mittel_30"][:2], rtol=1e-6), key
    # Extrapolation d = 1, 200 Wiederholungen (sortiert)
    z = S.extrapolation(1, REF["1"]["ns"], REF["1"]["verhaeltnis"], wdh=EXTRA["1"]["wdh"])
    assert z[0]["verhaeltnis"] == pytest.approx(EXTRA["1"]["zeilen"][0]["verhaeltnis"], rel=1e-6)
    # Wahrheit, d = 1
    w = S.zwei_gegen_wahrheit(1024, WAHR["wdh"], 2026)
    assert w["verhaeltnis"] == pytest.approx(WAHR["zeilen"][-1]["verhaeltnis"], rel=1e-6)
    # Minibatch: k = 8 und 16 bei d = 10 und d = 2 (verschoben und gleich), 2 000 Zufalls-Seeds
    for d in (2, 10):
        for form in ("gleich", "verschoben"):
            r = S.minibatch_reihe(d, form, MB["seeds"], 2026, ks=(8,))
            assert r["zeilen"][0]["mittel"] == pytest.approx(MB["reihen"][f"{d}_{form}"]["zeilen"][0]["mittel"], rel=1e-6), (d, form)
    # Streuung: erster Satz (Zufalls-Seed 3000) bei d = 1
    r = S.reihe("normal", 1, 0.0, S.NS_ALLE, 30, 3000)
    assert r["steigung_alle"] == pytest.approx(STREU["normal_1"]["alle"][0], abs=1e-6) and r["steigung_letzte3"] == pytest.approx(STREU["normal_1"]["letzte3"][0], abs=1e-6)


def test_sliced_tabelle_neu_gerechnet_als_band():
    t = SL.tabelle(2026)
    for k in ("rauschboden_w2", "voll_spearman", "voll_kendall", "pop_spearman", "pop_kendall"):
        assert t[k] == pytest.approx(SLICED[k], abs=1e-6), k
    for z, z0 in zip(t["zeilen"], SLICED["zeilen"]):
        assert z["spearman"] == pytest.approx(z0["spearman"], abs=1e-6) and z["kendall"] == pytest.approx(z0["kendall"], abs=1e-6)


# --- (4) Zahlen im Text der App ---------------------------------------------------------------------------------------------------------------

def test_zahlen_im_fest_geschriebenen_text_der_app():
    assert E.prozent(REF["10"]["verhaeltnis"][-1]) == "46 %"                          # Intro: „rund 46 % des mittleren Paarabstands“
    assert E.zahl(REF["10"]["steigung_alle"], 2) == "−0,11"                           # Intro und Grenzen: „bei d = 10 −0,11 gegen −0,1“
    assert "rund 46 % des mittleren Paarabstands" in APP and "bei $d = 10$ −0,11 gegen −0,1" in APP and "gemessen −0,11 gegen −0,1 der Schranke" in APP
    assert "Spearman 0,93" in APP and E.zahl(SLICED["pop_spearman"], 2) == "0,93"      # Abschnitt sliced
    assert "Spearman 1,000" in APP and E.zahl(SLICED["pop_spearman_familie"], 3) == "1,000"
    assert "0,316" in APP and E.zahl(1 / math.sqrt(10), 3) == "0,316"
    assert "wahres W2² = 4 + d · 0,25" in APP and W is not None and S.w2sq_minibatch_wahr(2, "verschoben") == 4.5 and S.w2sq_minibatch_wahr(10, "verschoben") == 6.5
    assert "Abstand 6" in APP and 2 * S.BLOCK_ABSTAND == 6.0


def test_kernfragen_der_app_nennen_die_zahlen_der_messdatei():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    texte = " ".join(i.value for i in at.info)
    assert f"{E.zahl(REF['1']['steigung_alle'], 2)} / {E.zahl(REF['2']['steigung_alle'], 2)} / {E.zahl(REF['5']['steigung_alle'], 2)} / {E.zahl(REF['10']['steigung_alle'], 2)} für d = 1 / 2 / 5 / 10" in texte
    assert f"{E.zahl(MB['reihen']['10_gleich']['zeilen'][-1]['mittel'], 2)} statt 0 (zufällige Paarung: 20" in texte
    assert f"{E.zahl(STRUKTUR['gerade_0.0']['steigung_alle_100'], 2)} (wie d = 1)" in texte


# --- (5) Konstanten, Kantenwortlaut, Testzahl, verbotene Wörter -------------------------------------------------------------------------------

def test_konstanten_der_readme():
    for text in ("Zufalls-Seed 2026", "n = 32 bis 1 024", "2 000 Zufalls-Seeds", "n = 300", "−3 e₁ und +3 e₁", "4,5 bei d = 2 und 6,5 bei d = 10", "1 209 600 Normalzahlen"):
        assert text in README, text
    assert C.DEFAULT_SEED == 2026 and S.NS_ALLE == (32, 64, 128, 256, 512, 1024) and MB["seeds"] == 2000 and SL.N_STICHPROBE == 300
    assert S.BLOCK_ABSTAND == 3.0 and S.MB_MU == 2.0 and S.MB_S == 1.5 and S.MB_KS == (8, 16, 32, 64, 128)
    assert 2 * sum(S.NS_ALLE) * 10 * 30 == 1_209_600


def test_testzahl_der_readme():
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    zahl = int(re.search(r"(\d+) tests? collected", r.stdout).group(1))
    assert f"- **Testzahl:** {zahl} Tests (`pytest --collect-only -q tests`)." in README, zahl


def test_readme_und_app_nennen_keine_planbegriffe_und_keine_umgebungsaussagen():
    for text in (README, APP):
        for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Saatwert", "Vormessung", "Mutationstest", "Stück", "V1", "O6", "TODO ", "Plan-", "PLAN-",
                         "wasserstein-demo", "Neuheit ist", "erstmals", "exponentiell"):
            if verboten == "TODO " and text is README:
                continue                                                # der Live-Link-Kommentar bleibt bis zum Deploy
            assert verboten not in text, verboten
        assert not re.search(r"\b(?:trägt|tragen)\b", text), "„trägt/tragen“ nicht verwenden („beträgt“ ist erlaubt)"
        assert not re.search(r"\bSeeds?\b", re.sub(r"Zufalls-Seeds?|Hash-Seed|Seed-Sätze", "", text)), "„Seed“ nur als „Zufalls-Seed“"
    assert "Orakel" not in APP


def test_readme_nennt_den_kantenwortlaut():
    assert "*Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1.*" in README
    assert "*Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1.*" in APP
    assert "zeigt Stichprobenrauschen nur an einem Beispiel (eine Dimension, ein wählbares n) und hebt die Annahme nicht auf" in README
    assert "zeigt Stichprobenrauschen nur an einem Beispiel (eine Dimension, ein wählbares n) und hebt die Annahme nicht auf" in APP


def test_befunde_der_aussagenpruefung_stehen_in_readme_und_app():
    """Einschränkungen aus der Prüfung vom 2026-11-24: Dreiecksungleichung, Abstand zur Wahrheit, Zellen mit Standardfehler, Rauschen der Geraden, sliced aus Stichproben, Quellen."""
    for text in (README, APP):
        assert "Dreiecksungleichung" in text
        assert "unbeschränkt" not in text and "dreißigmal" not in text
        assert "Spearman 0,980" in text and "Spearman 1,000" in text and "Spearman 0,93" in text
        assert "Volltext" in text and "nur Auszug" not in text and "Kap. 2, gelesen" not in text
    assert "(eine Stichprobe je Fall, ohne Streuungsangabe)" in APP and "(eine Stichprobe je Fall, ohne Streuungsangabe)" in README
    assert "wird nur bei d = 1 berechnet" in README and "die Demo rechnet ihn dort nicht" in README and "die Demo rechnet den Abstand zur Wahrheit dort nicht" in APP
    assert "über 100-mal langsamer" in APP and "mehr als 700-mal" in README and "107- bis 167-mal" in README
    # Sliced aus Stichproben: Spearman je Familie bei L = 500 ist 0,980 (README und App nennen es)
    assert E.zahl(SLICED["zeilen"][-1]["spearman_familie"], 3) == "0,980" and SLICED["zeilen"][-1]["L"] == 500
    # Standardfehler der Zellen: je Dimension und n vorhanden, größte bei n = 32 unter einem Prozentpunkt, das Beispiel d = 1, n = 32 liegt 2 bis 3 Standardfehler über der Monte-Carlo-Erwartung
    assert all(len(REF[d]["se_verhaeltnis"]) == 6 for d in REF) and max(REF[d]["se_verhaeltnis"][0] for d in REF) < 0.011
    z, se = REF["1"]["verhaeltnis"][0], REF["1"]["se_verhaeltnis"][0]
    assert 2.0 < (z - R.MC_ERWARTUNG_D1_N32) / se < 3.0 and "rund 2,4 Standardfehler über der Erwartung 26,8 %" in README
    # Gerade in R^10: der Wert −0,37 liegt rund zwei Standardabweichungen neben dem Mittel der zehn Sätze
    abw = abs(STRUKTUR["gerade_0.0"]["steigung_letzte3_30"] - STREU["gerade_10"]["letzte3_mittel"]) / STREU["gerade_10"]["letzte3_std"]
    assert 1.8 < abw < 2.4 and "mit dem Rauschen vereinbar" in README and "ist Rauschen, kein Gegenbeispiel" not in README
    assert "um −0,5 (bei 30 Wiederholungen mit einer Streuung von etwa ±0,1" in C.PRESET_HELP["Gerade in R^10"]
    # Fazit Experiment 3: die Zahlen je Familie stehen in der Messdatei
    g = [STRUKTUR[f"gerade_{s}"]["steigung_alle_100"] for s in ("0.1", "0.3")]
    b = [STRUKTUR[f"bloecke_{s}"]["steigung_alle_100"] for s in ("0.1", "0.3")]
    assert -0.25 < g[0] < -0.15 and -0.16 < g[1] < -0.10 and -0.36 < b[0] < -0.30 and -0.24 < b[1] < -0.18


def test_monte_carlo_erwartung_der_zelle_d1_n32_als_band():
    """Die Erwartung 26,8 % (W1 durch Maßstab, d = 1, n = 32) aus 100 000 Wiederholungen mit numpy-Zufall; Standardfehler 0,03 Prozentpunkte, Band ± 0,3."""
    rng = np.random.default_rng(12345)
    x = np.sort(rng.standard_normal((100_000, 32)), axis=1)
    y = np.sort(rng.standard_normal((100_000, 32)), axis=1)
    erwartung = float((np.abs(x - y).mean(axis=1) / W.mittlerer_paarabstand(1)).mean())
    assert abs(erwartung - R.MC_ERWARTUNG_D1_N32) < 0.003


def test_die_formel_n_hoch_minus_1_durch_d_steht_nur_mit_ab_d_3():
    """Die Formel r ∝ n^(−1/d) darf nie ohne „ab d = 3“ (oder „ab d ≥ 3“) stehen (bei d = 1 gemessen −0,49, nicht −1)."""
    for name, text in (("README", README), ("app.py", APP)):
        for m in re.finditer(r"n\^\{?\(?−1/d|n\^\{-1/d\}", text):
            umgebung = text[max(0, m.start() - 160): m.end() + 160]
            assert re.search(r"\$?d (=|≥|\\ge|\\geq) ?3", umgebung), (name, umgebung)


def test_scipy_steht_in_den_requirements_und_nur_die_zuordnung_wird_importiert():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scipy" in req
    for path in sorted(ROOT.glob("*.py")):
        for zeile in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*(from|import)\s+scipy", zeile):
                assert zeile.strip() == "from scipy.optimize import linear_sum_assignment as _lsa", (path.name, zeile)
