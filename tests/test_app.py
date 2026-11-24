"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jedes Preset, fährt jeden Regler an seine Grenzen,
prüft den Permalink, die Verdikt-Zeile je Preset und einen Worst-Case mit allen Maxima; kein Python-Fehler, insbesondere kein `StreamlitDuplicateElementId`
(mehrere st.plotly_chart ohne eindeutiges key=) und keine Ausnahme bei Reglern am Rand."""

import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dbd_constants as C
import dbd_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio")


def _frisch(**params) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in params.items():
        at.query_params[k] = str(v)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _texte(at):
    return [e.value for e in list(at.info) + list(at.warning)]


def _verdikt(at, anfang):
    treffer = [t for t in _texte(at) if t.startswith(anfang)]
    assert treffer, (anfang, [t[:60] for t in _texte(at)])
    return treffer[0]


def test_app_laedt_ohne_ausnahme_und_zeigt_drei_experimente_und_vier_diagramme():
    at = _frisch()
    assert len(at.get("plotly_chart")) == 4          # Experiment 1, zwei Diagramme in Experiment 2 (Zwei Stichproben), Experiment 3
    uebersicht = [s.value for s in at.subheader]
    assert [u[:14] for u in uebersicht if u.startswith("🔬")] == ["🔬 Experiment 1", "🔬 Experiment 2", "🔬 Experiment 3"]
    assert any("Wo die Annahmen enden" in u for u in uebersicht)
    assert not at.error


def test_jedes_preset_laeuft_ohne_ausnahme_durch():
    at = _frisch()
    for name in C.PRESETS:
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


def test_preset_d_gleich_1_meldet_nicht_minus_eins_sondern_etwa_minus_ein_halb():
    at = _frisch()
    at.sidebar.button(key="preset_d = 1").click().run(timeout=TIMEOUT)
    text = _verdikt(at, "d = 1: Steigung")
    m = re.search(r"Steigung (−0,\d\d) zwischen n = 32", text)
    assert m and -0.55 < float(m.group(1).replace("−", "-").replace(",", ".")) < -0.45
    assert "nicht −1" in text


def test_preset_d_gleich_10_meldet_die_flache_steigung_und_den_rauschboden():
    at = _frisch()
    at.sidebar.button(key="preset_d = 10").click().run(timeout=TIMEOUT)
    text = _verdikt(at, "d = 10: Steigung")
    m = re.search(r"Steigung (−0,\d\d) zwischen n = 32", text)
    assert m and -0.14 < float(m.group(1).replace("−", "-").replace(",", ".")) < -0.09
    assert "46 %" in text or "45 %" in text or "47 %" in text
    assert "n ≈ 7,1e8*" in text and "n ≈ 3,3e11*" in text and "bis zu 8,5 Größenordnungen" in text      # nicht 5,8: die größte gesternte Zahl zählt
    assert any(w.value.startswith("d = 10: Steigung") for w in at.warning)


def test_preset_gerade_in_r10_meldet_keine_dimensionssteigung():
    at = _frisch()
    at.sidebar.button(key="preset_Gerade in R^10").click().run(timeout=TIMEOUT)
    text = _verdikt(at, "R^10")
    m = re.search(r"Steigung (−0,\d\d) \(alle n\)", text)
    assert m and -0.6 < float(m.group(1).replace("−", "-").replace(",", ".")) < -0.35
    assert "wie d = 1 (um −0,5; einzelne Werte weichen" in text and "Gerade:" in text and "Standardabweichung von 0,08 bei d = 1 und 0,06 bei der Geraden" in text


def test_preset_zwei_atome_meldet_minus_ein_halb_exakt():
    at = _frisch()
    at.sidebar.button(key="preset_Zwei Atome").click().run(timeout=TIMEOUT)
    text = _verdikt(at, "R^10")
    assert "zwei Atome im Abstand 6" in text and "exakt −0,499" in text and "die Dimension geht nicht ein" in text
    assert any("Zwei Atome, exakt" in m.value for m in at.markdown)


def test_preset_minibatch_zeigt_den_minibatch_teil_mit_verdikt():
    at = _frisch()
    at.sidebar.button(key="preset_Minibatch d = 10").click().run(timeout=TIMEOUT)
    text = _verdikt(at, "d = 10, k = 128")
    assert "V = Minibatch-Wert" in text and "wahres W2²" in text and "im Mittel liegt V nie darunter" in text
    assert [c for c in at.get("plotly_chart")]


@pytest.mark.parametrize("form", C.FORMM)
@pytest.mark.parametrize("dm", C.DM)
def test_minibatch_in_beiden_formen_und_dimensionen(form, dm):
    at = _frisch(m2=C.MODUS2[1], dm=dm, fm=form, sm=200)
    assert _verdikt(at, f"d = {dm}, k = 128")


@pytest.mark.parametrize("art", C.ART3)
def test_jede_datenart_in_experiment_3(art):
    at = _frisch(a3=art)
    assert _verdikt(at, "R^10")


@pytest.mark.parametrize("d", C.D1)
def test_jede_dimension_in_experiment_1(d):
    at = _frisch(d1=d, nm=256, w1=10)
    assert _verdikt(at, f"d = {d}: Steigung")


def test_permalink_uebernimmt_werte_begrenzt_und_rastet_ein():
    at = _frisch(d1=5, nm=100000, w1=40, m2=C.MODUS2[1], km=16, s3="0.12", seed=7)
    assert _regler(at, "d1").value == 5 and _regler(at, "nmax1").value == 1024 and _regler(at, "wdh1").value == 30
    assert _regler(at, "modus2").value == C.MODUS2[1] and _regler(at, "km").value == 16 and _regler(at, "sigma3").value == 0.1 and _regler(at, "seed_input").value == 7
    assert at.query_params["d1"] == ["5"] or at.query_params["d1"] == "5"


def test_permalink_ungueltige_werte_werden_ignoriert():
    at = _frisch(d1="abc", m2="irgendwas", seed="x")
    assert _regler(at, "d1").value == C.DEFAULTS["d1"] and _regler(at, "modus2").value == C.MODUS2[0] and _regler(at, "seed_input").value == C.DEFAULT_SEED


def test_jeder_regler_am_minimum_und_am_maximum_ohne_ausnahme():
    """Ein AppTest-Lauf, Regler nacheinander (erst Maximum, dann Minimum, damit die teuren Einstellungen nicht stehen bleiben); die Minibatch-Regler erst nach dem Umschalten.
    Die 100 Wiederholungen von Experiment 3 an den Randwerten rechnet der Worst-Case-Test unten (hier nur das Minimum: sonst kostet jede Änderung der Datenart 11 Sekunden)."""
    at = _frisch(w3=10)
    for key, spec in P.SETTING_SPECS.items():
        if key in ("dm", "formm", "seedsm", "km"):
            continue
        werte = list(spec.options) if spec.options is not None else [spec.lo, spec.hi]
        for wert in ((werte[-1], werte[0]) if key != "wdh3" else (werte[0],)):
            if key in ("n2", "d2", "wdh2") and _regler(at, "modus2").value != C.MODUS2[0]:
                _regler(at, "modus2").set_value(C.MODUS2[0]).run(timeout=TIMEOUT)
            if key == "sigma3" and _regler(at, "art3").value not in (C.ART3[1], C.ART3[3]):
                _regler(at, "art3").set_value(C.ART3[1]).run(timeout=TIMEOUT)         # σ ist bei Gerade und Zwei Atomen deaktiviert
            _regler(at, key).set_value(wert).run(timeout=TIMEOUT)
            assert not at.exception, (key, wert, [str(e) for e in at.exception])
    _regler(at, "modus2").set_value(C.MODUS2[1]).run(timeout=TIMEOUT)
    for key in ("dm", "formm", "seedsm", "km"):
        werte = list(P.SETTING_SPECS[key].options)
        for wert in (werte[-1], werte[0]):
            _regler(at, key).set_value(wert).run(timeout=TIMEOUT)
            assert not at.exception, (key, wert, [str(e) for e in at.exception])


def test_worst_case_alle_maxima_zugleich_in_beiden_teilen():
    """Größtes n, 100 Wiederholungen, d = 10, 2 000 Zufalls-Seeds im Minibatch: kein Zeitlimit-Überschreiten (AppTest-Limit), keine Ausnahme."""
    at = _frisch(d1=10, nm=1024, w1=100, n2=1024, d2=10, w2=100, a3=C.ART3[1], s3="1.0", w3=100)
    assert _verdikt(at, "d = 10: Steigung")
    _regler(at, "modus2").set_value(C.MODUS2[1]).run(timeout=TIMEOUT)
    _regler(at, "seedsm").set_value(2000).run(timeout=TIMEOUT)
    _regler(at, "formm").set_value(C.FORMM[1]).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert _verdikt(at, "d = 10, k = 128")


def test_rueckfall_ohne_scipy_gibt_dieselben_zahlen_und_einen_hinweis(monkeypatch):
    """Ohne scipy rechnet die eigene Ungarische Methode: dieselben Verdikt-Zahlen, dazu der Hinweis auf den langsameren Rückfall (kleine Einstellung, sonst dauert er Minuten)."""
    import streamlit as st

    import dbd_wasserstein as W
    params = dict(d1=10, nm=256, w1=10, n2=64, d2=5, w2=10, a3=C.ART3[1], w3=10)
    st.cache_data.clear()
    try:
        mit = _frisch(**params)
        assert not any("scipy ist nicht installiert" in w.value for w in mit.warning)
        text_mit = (_verdikt(mit, "d = 10: Steigung"), _verdikt(mit, "R^10"))
        st.cache_data.clear()
        monkeypatch.setattr(W, "_lsa", None)
        ohne = _frisch(**params)
        assert any("scipy ist nicht installiert" in w.value for w in ohne.warning)
        assert (_verdikt(ohne, "d = 10: Steigung"), _verdikt(ohne, "R^10")) == text_mit
    finally:
        st.cache_data.clear()
