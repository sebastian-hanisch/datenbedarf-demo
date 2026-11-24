"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert. Regler, die der Umschalter „Teil“
in Experiment 2 deaktiviert, werden erst nach dem Umschalten geändert (AppTest kann einen deaktivierten Regler nicht setzen)."""

from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import dbd_constants as C
import dbd_presets as P
import dbd_sweeps as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")
NUR_MINIBATCH = ("dm", "formm", "seedsm", "km")        # deaktiviert, solange „Zwei Stichproben“ gewählt ist


def _schnell_w1(familie, d, sigma, n, wdh, seed, verfahren="auto"):
    """Ersatz für die Zuordnungen: gleichmäßig fallende Werte, nur damit die Oberfläche in Millisekunden steht (hier geht es um Regler, nicht um Zahlen)."""
    return np.full(wdh, 1.0 / n ** 0.5) * (1.0 + 0.01 * np.arange(wdh))


def _schnell_minibatch(d, k, seeds, seed, form, verfahren="auto"):
    return np.full(seeds, (6.5 if form == "verschoben" else 1.0) * (1.0 + 8.0 / k)) * (1.0 + 0.001 * np.arange(seeds))


@pytest.fixture(autouse=True)
def schnelle_rechnung(monkeypatch):
    """Die Layout-Tests prüfen Regler und Hervorhebung; die Rechnung wird ersetzt und der Zwischenspeicher davor und danach geleert (sonst sähen andere Tests die Ersatzzahlen)."""
    st.cache_data.clear()
    monkeypatch.setattr(S, "w1_wiederholungen", _schnell_w1)
    monkeypatch.setattr(S, "minibatch_werte", _schnell_minibatch)
    yield
    st.cache_data.clear()


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _anderer_wert(regler, spec):
    """Ein Wert, der vom aktuellen verschieden ist (Auswahllisten: die erste abweichende Option)."""
    if spec.options is not None:
        return next(o for o in spec.options if o != regler.value)
    return spec.hi if regler.value != spec.hi else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_ein_preset_setzt_auch_die_nicht_genannten_regler_auf_den_standard_zurueck():
    at = _app()
    at.sidebar.select_slider(key="d1").set_value(5).run()
    at.sidebar.button(key="preset_Zwei Atome").click().run()
    assert at.sidebar.select_slider(key="d1").value == C.DEFAULTS["d1"]
    assert at.sidebar.radio(key="art3").value == C.ART3[2]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel, "ein Eintrag in SETTING_SPECS hat keinen Regler"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    if state_key in NUR_MINIBATCH:
        at.sidebar.radio(key="modus2").set_value(C.MODUS2[1]).run()
        assert not at.exception, [str(e) for e in at.exception]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(regler, spec)).run()
    assert not at.exception, [str(e) for e in at.exception]
    name = next(iter(C.PRESETS))
    assert _typen(at)[name] != "primary"          # das Standard-Preset ist nicht mehr hervorgehoben (ein anderes Preset darf es sein: „Minibatch d = 10“ ist genau der Umschalter)
    assert list(_typen(at).values()).count("primary") <= 1
    # erneutes Anklicken markiert wieder
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_breite_tabellen_scrollen_waagerecht_statt_die_seite_zu_verschieben():
    """Telefonbreite (375 px): Markdown-Tabellen bekommen eine eigene waagerechte Bildlaufleiste (Stylesheet gleich nach dem Kopf in app.py)."""
    quelle = Path(APP).read_text(encoding="utf-8")
    regel = '<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>'
    assert f"st.html('{regel}')" in quelle
    assert quelle.index("st.html(kopf_css(dunkles_design()))") < quelle.index(regel)
