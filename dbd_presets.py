"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler hat einen Eintrag in SETTING_SPECS (Name im Session State = Schlüssel in `dbd_constants.DEFAULTS`). Reglerwerte aus der Adresszeile werden auf die Grenzen begrenzt
beziehungsweise auf den nächsten erlaubten Wert eingerastet (Auswahllisten); ungültige Werte werden ignoriert. Ein Preset legt die Regler fest, die es nennt; alle anderen
gehen beim Anklicken auf ihren Standardwert zurück."""

import random
from dataclasses import dataclass

import streamlit as st

import dbd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # erlaubte Werte (Auswahllisten und Stufen)


def _opt(url, caster, key, options):
    return SettingSpec(url, caster, C.DEFAULTS[key], options=options)


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULTS["seed_input"], 0, C.SEED_MAX),
    "d1": _opt("d1", int, "d1", C.D1),
    "nmax1": _opt("nm", int, "nmax1", C.NMAX1),
    "wdh1": _opt("w1", int, "wdh1", C.WDH1),
    "modus2": _opt("m2", str, "modus2", C.MODUS2),
    "n2": _opt("n2", int, "n2", C.N2),
    "d2": _opt("d2", int, "d2", C.D2),
    "wdh2": _opt("w2", int, "wdh2", C.WDH2),
    "dm": _opt("dm", int, "dm", C.DM),
    "formm": _opt("fm", str, "formm", C.FORMM),
    "seedsm": _opt("sm", int, "seedsm", C.SEEDSM),
    "km": _opt("km", int, "km", C.KM),
    "art3": _opt("a3", str, "art3", C.ART3),
    "sigma3": _opt("s3", float, "sigma3", C.SIGMA3),
    "wdh3": _opt("w3", int, "wdh3", C.WDH3),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _eingerastet(spec: SettingSpec, wert):
    """Wert auf die Grenzen begrenzt beziehungsweise auf die nächste erlaubte Option eingerastet; None, wenn nichts passt."""
    if spec.options is not None:
        if spec.caster is str:
            return wert if wert in spec.options else None
        if not isinstance(wert, (int, float)) or wert != wert:
            return None
        return min(spec.options, key=lambda o: abs(o - float(wert)))
    if isinstance(wert, float) and wert != wert:
        return None
    if spec.lo is not None:
        wert = max(spec.lo, wert)
    if spec.hi is not None:
        wert = min(spec.hi, wert)
    return wert


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (begrenzt beziehungsweise eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                wert = _eingerastet(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if wert is not None:
                st.session_state[state_key] = wert
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler setzen: der Wert des Presets, sonst der Standardwert."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
