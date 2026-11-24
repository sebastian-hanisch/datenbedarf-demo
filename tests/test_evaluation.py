"""Formatierung und Verdikte an von Hand gesetzten Werten: Schwellen, Zweige und jede Zahl im Text (DEMO-PLAYBOOK §12: Verdikte sind reine Funktionen)."""

import re

import numpy as np
import pytest

import dbd_constants as C
import dbd_evaluation as E
import dbd_sweeps as S


# --- Formatierung -----------------------------------------------------------------------------------------------------------------------------

def test_zahl_mit_dezimalkomma_minuszeichen_und_ohne_minus_null():
    assert E.zahl(0.4568, 3) == "0,457" and E.zahl(-0.113, 2) == "−0,11" and E.zahl(-0.0001, 2) == "0,00"
    assert E.zahl(float("nan")) == "–" and E.zahl(float("inf")) == "∞" and E.zahl(float("-inf")) == "−∞"


def test_ganz_und_prozent():
    assert E.ganz(1024) == "1 024" and E.ganz(2000) == "2 000" and E.ganz(7) == "7"
    assert E.prozent(0.4568) == "46 %" and E.prozent(0.0494, 1) == "4,9 %"


def test_gross_n_schreibt_bis_99999_aus_und_darueber_in_zehnerpotenzen():
    assert E.gross_n(686.4) == "686" and E.gross_n(3720.4) == "3 720" and E.gross_n(1038.0) == "1 040" and E.gross_n(23300.0) == "23 300"
    assert E.gross_n(7.08e8) == "7,1e8" and E.gross_n(3.28e11) == "3,3e11" and E.gross_n(1.46e5) == "1,5e5"


def test_exp_text():
    assert E.exp_text(1.2345e-12) == "1,2e-12" and E.exp_text(0.0) == "0" and E.exp_text(2.5e8, 2) == "2,50e8"


def test_tabelle_md_kopf_trenner_und_zeilen():
    t = E.tabelle_md(["a", "b"], [["1", "2"], ["3", "4"]]).split("\n")
    assert t == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_erwartung_nach_fournier_guillin_je_dimension():
    assert E.erwartung_text(1) == "−1/2" and E.erwartung_text(2) == "−1/2 mit Log-Faktor" and E.erwartung_text(3) == "−1/3" and E.erwartung_text(10) == "−1/10"


def test_halbierungsfaktor_aus_der_steigung_von_hand():
    """2^(1/0,199) = 32,6; 2^(1/0,112) = 487,5 (Rechnung)."""
    assert E.halbierungsfaktor(-0.199) == pytest.approx(2 ** (1 / 0.199), rel=1e-12) and 32.5 < E.halbierungsfaktor(-0.199) < 32.8
    assert 486 < E.halbierungsfaktor(-0.112) < 489


# --- Einstellungen -----------------------------------------------------------------------------------------------------------------------------

def test_settings_aus_zustand_nimmt_fehlende_schluessel_als_standard():
    s = E.settings_aus_zustand({"d1": 5})
    assert s.d1 == 5 and s.seed_input == C.DEFAULT_SEED and s.art3 == C.ART3[1]
    assert E.settings_aus_zustand({}) == E.Settings()


def test_familie_sigma_je_datenart():
    s = E.Settings(sigma3=0.3)
    assert E.familie_sigma(E.Settings(art3=C.ART3[0], sigma3=0.3)) == ("gerade", 0.0)
    assert E.familie_sigma(E.Settings(art3=C.ART3[1], sigma3=0.3)) == ("gerade", 0.3)
    assert E.familie_sigma(E.Settings(art3=C.ART3[2], sigma3=0.3)) == ("bloecke", 0.0)
    assert E.familie_sigma(E.Settings(art3=C.ART3[3], sigma3=0.3)) == ("bloecke", 0.3)
    assert s.sigma3 == 0.3


def test_exakte_steigung_der_zwei_atome_ist_minus_0_499():
    assert E.exakte_steigung_atome() == pytest.approx(-0.499, abs=0.0006)
    assert E.exakte_steigung_atome() != pytest.approx(-0.5, abs=1e-4)


# --- Verdikt 1 -------------------------------------------------------------------------------------------------------------------------------

def _r1(d, steig, anteil, noetig10, noetig5, ex=None, wdh=30, n_max=1024):
    ns = S.ns_bis(n_max)
    return {
        "d": d, "ns": ns, "wdh": wdh, "steigung_alle": steig, "steigung_letzte3": steig - 0.01, "massstab": 1.13,
        "verhaeltnis": np.linspace(0.3, anteil, len(ns)), "noetig": {0.10: noetig10, 0.05: noetig5}, "extrapolation": ex,
    }


def test_verdikt1_d_gleich_1_nennt_steigung_anteil_nur_innerhalb_des_rasters_und_nicht_minus_eins():
    s = E.Settings(seed_input=C.DEFAULT_SEED)
    kind, text = E.verdikt1(s, _r1(1, -0.49, 0.05, (256.0, False), (900.0, False)))
    assert kind == "info"
    assert "Steigung −0,49 zwischen n = 32 und 1 024" in text and "über die letzten drei n −0,50" in text and "5 % des mittleren Paarabstands" in text
    assert "genügt n ≈ 256" in text and "n ≈ 900" in text and "*" not in text and "innerhalb des gemessenen Bereichs" in text
    assert "etwa −0,5, nicht −1" in text and "Erwartung der oberen Schranke: −1/2" in text and "30 Wiederholungen, Zufalls-Seed 2026" in text


def test_verdikt1_d_gleich_10_warnt_und_kennzeichnet_die_extrapolation_mit_stern_und_groessenordnungen():
    ex = {"wdh": 10, "zeilen": [{"verhaeltnis": 1.006}, {"verhaeltnis": 1.009}]}
    kind, text = E.verdikt1(E.Settings(), _r1(10, -0.113, 0.457, (7.08e8, True), (3.28e11, True), ex))
    assert kind == "warning"
    assert "n ≈ 7,1e8*" in text and "n ≈ 3,3e11*" in text and "bis zu 8,5 Größenordnungen" in text and "46 % des mittleren Paarabstands" in text
    assert "1,01 und 1,01 der Vorhersage" in text and "10 Wiederholungen" in text
    assert "Die gemessene Steigung ist flacher als −1/2 (bei d = 10: −0,11)" in text and "des Rauschbodens (46 % des Paarabstands bei n = 1 024)" in text and "nicht als Abstand der Verteilungen zu lesen" in text
    assert "Erwartung der oberen Schranke: −1/10" in text


def test_verdikt1_nennt_die_faktor_4_messung_nur_fuer_die_standardeinstellung():
    ex = {"wdh": 10, "zeilen": [{"verhaeltnis": 0.949}, {"verhaeltnis": 0.974}]}
    r = _r1(2, -0.42, 0.084, (686.0, False), (3540.0, True), ex)
    assert "0,95 und 0,97 der Vorhersage" in E.verdikt1(E.Settings(), r)[1]
    assert "Vorhersage" not in E.verdikt1(E.Settings(seed_input=7), r)[1]
    assert "Vorhersage" not in E.verdikt1(E.Settings(), _r1(2, -0.42, 0.084, (686.0, False), (3540.0, True), ex, wdh=10))[1]
    assert "Vorhersage" not in E.verdikt1(E.Settings(), _r1(2, -0.42, 0.084, (686.0, False), (3540.0, True), ex, n_max=256))[1]


def test_verdikt1_ohne_letzte_drei_n_bei_kleinem_raster_nur_die_steigung_ueber_alle():
    r = _r1(2, -0.4, 0.2, (100.0, False), (300.0, False), n_max=64)
    r["ns"] = (32, 64)
    assert "über die letzten drei n" not in E.verdikt1(E.Settings(), r)[1]


# --- Verdikt 2 -------------------------------------------------------------------------------------------------------------------------------

def _a2(verhaeltnisse, d2=1):
    tab = [{"n": n, "zwei": 0.2, "wahrheit": 0.14, "verhaeltnis": v} for n, v in zip(S.NS_ALLE, verhaeltnisse)]
    return {"n": 256, "d": d2, "wdh": 100, "anteil": 0.4568, "tabelle_d1": tab,
            "zwei_punkt": [{"N": N, "zwei": 0.0, "wahr": 0.0, "verhaeltnis": v} for N, v in ((2, 1.5), (4, 1.4583), (32, 1.4197), (1024, 1.4144))]}


def test_verdikt2_zwei_nennt_mittel_spanne_und_zwei_punkt_werte():
    kind, text = E.verdikt2_zwei(E.Settings(), _a2([1.38, 1.35, 1.42, 1.38, 1.41, 1.44]))
    assert kind == "info"
    assert "beim 1,40-Fachen" in text and "zwischen 1,35 und 1,44" in text and "100 Wiederholungen" in text
    assert "exakt 1,5 bei N = 2" in text and "1,414 bei N = 1 024" in text and "√2 ≈ 1,41" in text
    assert "nicht gemessen" not in text


def test_verdikt2_zwei_sagt_bei_d_ab_2_dass_der_faktor_nicht_gemessen_ist():
    text = E.verdikt2_zwei(E.Settings(d2=10, n2=1024), _a2([1.4] * 6, d2=10))[1]
    assert "Für d ≥ 2 ist dieser Faktor nicht gemessen" in text and "d = 10 und n = 1 024" in text and "46 % des mittleren Paarabstands" in text


def _a_mb(form, d, v, se, anteil, mittel=6.66, zwei=None):
    zeilen = [{"k": k, "mittel": mittel, "se": 0.0123, "v": v, "v_se": se, "anteil_unter_1": anteil} for k in S.MB_KS]
    r = {"d": d, "form": form, "seeds": 2000, "wahr": 6.5 if d == 10 else 4.5, "zufallspaarung": 2.0 * d if form == "gleich" else float("nan"), "zeilen": zeilen}
    if zwei:
        r["zwei_stichproben"] = [{"k": k, "mittel": zwei, "se": 0.02, "wdh": 500} for k in S.MB_KS]
    return r


def test_verdikt2_minibatch_gleich_nennt_wert_zufallspaarung_und_den_zwei_stichproben_abstand():
    kind, text = E.verdikt2_mb(E.Settings(km=128), _a_mb("gleich", 10, float("nan"), float("nan"), float("nan"), zwei=6.65))
    assert kind == "info"
    assert "d = 10, k = 128" in text and "6,66 ± 0,01 statt 0" in text and "2 000 Zufalls-Seeds" in text and "Zufallspaarung kostet 20" in text
    assert "derselbe Rauschboden" in text and "6,65 ± 0,02" in text and "500 Wiederholungen" in text


def test_verdikt2_minibatch_verschoben_nennt_v_anteil_und_warnt_bei_d_10():
    kind, text = E.verdikt2_mb(E.Settings(km=32), _a_mb("verschoben", 10, 3.1252, 0.0063, 0.0))
    assert kind == "warning"
    assert "V = Minibatch-Wert durch wahres W2² (6,5) = 3,125 ± 0,006" in text and "0 % der einzelnen Batches" in text and "im Mittel liegt V nie darunter" in text and "in allen gemessenen Zellen so" in text
    assert "Steigung von log(V − 1)" not in text


def test_verdikt2_minibatch_bei_d_2_nennt_den_anteil_unter_eins_und_die_streuung_der_steigung():
    kind, text = E.verdikt2_mb(E.Settings(km=128, dm=2), _a_mb("verschoben", 2, 1.0571, 0.0033, 0.368))
    assert kind == "info"
    assert "(4,5) = 1,057 ± 0,003" in text and "37 % der einzelnen Batches" in text and "Steigung von log(V − 1)" in text


# --- Verdikt 3 -------------------------------------------------------------------------------------------------------------------------------

def _r3(familie, sigma, steig=-0.17, exakt=None):
    r = {"familie": familie, "sigma": sigma, "wdh": 30, "steigung_alle": steig, "steigung_letzte3": steig + 0.03, "exakt": exakt,
         "volle_dimension": {"steigung_alle": -0.113, "wdh": 30}}
    if exakt is None:
        del r["exakt"]
    return r


def test_verdikt3_gerade_sagt_wie_d_gleich_1():
    kind, text = E.verdikt3(E.Settings(art3=C.ART3[0]), _r3("gerade", 0.0, -0.48))
    assert kind == "info"
    assert "Gerade: Steigung −0,48 (alle n), −0,45 (letzte drei n)" in text and "wie d = 1 (um −0,5; einzelne Werte weichen" in text and "Standardnormal in R^10, 30 Wiederholungen, Zufalls-Seed 2026): −0,11" in text
    assert E.streuung_text() in text and "Standardabweichung von 0,08 bei d = 1 und 0,06 bei der Geraden in R^10" in text and "keine bessere Rate" in text


def test_verdikt3_gerade_mit_rauschen_nennt_sigma_und_beide_steigungen():
    text = E.verdikt3(E.Settings(art3=C.ART3[1], sigma3=0.3), _r3("gerade", 0.3, -0.12))[1]
    assert "Gerade + Rauschen: mit Streuung σ = 0,3 auf den neun Nebenachsen: Steigung −0,12 (alle n), −0,09 (letzte drei n)" in text


def test_verdikt3_zwei_atome_nennen_gemessen_und_exakt():
    text = E.verdikt3(E.Settings(art3=C.ART3[2]), _r3("bloecke", 0.0, -0.55, exakt=-0.4993))[1]
    assert "zwei Atome im Abstand 6: Steigung −0,55 (alle n) gemessen, exakt −0,499 aus der Binomialrechnung" in text and "die Dimension geht nicht ein" in text


def test_verdikt3_zwei_bloecke_nennen_sigma():
    text = E.verdikt3(E.Settings(art3=C.ART3[3], sigma3=0.1), _r3("bloecke", 0.1, -0.3))[1]
    assert "zwei Blöcke mit Streuung σ = 0,1 je Achse: Steigung −0,30 (alle n), −0,27 (letzte drei n)" in text


def test_verdikt3_ohne_bezugsreihe_laesst_den_vergleich_weg():
    r = _r3("gerade", 0.1)
    r["volle_dimension"] = None
    assert "Zum Vergleich" not in E.verdikt3(E.Settings(), r)[1]


def test_keine_planbegriffe_und_kein_traegt_in_den_verdikten():
    texte = [
        E.verdikt1(E.Settings(), _r1(10, -0.113, 0.457, (7.08e8, True), (3.28e11, True)))[1],
        E.verdikt2_zwei(E.Settings(), _a2([1.4] * 6))[1],
        E.verdikt2_mb(E.Settings(), _a_mb("verschoben", 2, 1.06, 0.003, 0.3))[1],
        E.verdikt3(E.Settings(), _r3("gerade", 0.1))[1],
    ]
    for t in texte:
        for verboten in ("Phase 0", "Pilot", "trägt", "tragen", "Orakel", "Saatwert", "V1", "H 3"):
            assert verboten not in t
        assert not re.search(r"\d\.\d", t), t          # Dezimalkomma statt Dezimalpunkt


def test_verdikt1_groessenordnungen_nehmen_die_groesste_gesternte_zahl():
    """Fehlerfall (Prüfung 2026-11-24): bei d = 10 stand „bis zu 5,8“ (nur die 10-%-Zahl 7,1e8) statt 8,5 (die 5-%-Zahl 3,3e11); ebenso bei d = 3, 4, 5. Je Dimension der Bezugsreihe."""
    import json
    from pathlib import Path
    ref = json.loads((Path(__file__).resolve().parent.parent / "tools" / "messreihen.json").read_text(encoding="utf-8"))["ref"]
    erwartet = {"1": "0,0", "2": "0,5", "3": "1,5", "4": "2,6", "5": "3,7", "10": "8,5"}
    for d, text_gr in erwartet.items():
        r = ref[d]
        n10, st10 = S.noetiges_n(r["ns"], r["verhaeltnis"], 0.10)
        n5, st5 = S.noetiges_n(r["ns"], r["verhaeltnis"], 0.05)
        assert st5, d
        unabhaengig = max(np.log10(n / 1024.0) for n, st in ((n10, st10), (n5, st5)) if st)
        assert E.zahl(unabhaengig, 1) == text_gr, d
        text = E.verdikt1(E.Settings(d1=int(d)), _r1(int(d), r["steigung_alle"], r["verhaeltnis"][-1], (n10, st10), (n5, st5)))[1]
        assert f"bis zu {text_gr} Größenordnungen" in text, (d, text)
    # nur die 10-%-Zahl gesternt: die Zahl gehört zu ihr; keine Sternzahl: kein Satz über Größenordnungen
    assert "bis zu 1,0 Größenordnungen" in E.verdikt1(E.Settings(), _r1(3, -0.3, 0.14, (10240.0, True), (900.0, False)))[1]
    assert "Größenordnungen" not in E.verdikt1(E.Settings(), _r1(1, -0.49, 0.05, (256.0, False), (900.0, False)))[1]
