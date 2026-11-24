"""Presets: Vollständigkeit, gültige Werte, Standard-Preset gleich Standardeinstellungen, paarweise verschieden; Permalink (Begrenzen, Einrasten)."""

import pytest

import dbd_constants as C
import dbd_evaluation as ev
import dbd_presets as P


def _voll(name):
    """Alle Regler eines Presets mit Standardwerten für die nicht genannten."""
    return {k: C.PRESETS[name].get(k, spec.default) for k, spec in P.SETTING_SPECS.items()}


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[name] for name in C.PRESETS)


def test_preset_keys_are_known_settings_and_values_are_allowed_options():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        for k, v in preset.items():
            spec = P.SETTING_SPECS[k]
            if spec.options is not None:
                assert v in spec.options, (name, k, v)
            else:
                assert spec.lo <= v <= spec.hi, (name, k, v)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert _voll(first) == {k: spec.default for k, spec in P.SETTING_SPECS.items()}
    assert ev.settings_aus_zustand(_voll(first)) == ev.Settings()


def test_presets_are_pairwise_different_settings():
    voll = {name: tuple(sorted(_voll(name).items())) for name in C.PRESETS}
    assert len(set(voll.values())) == len(C.PRESETS)


def test_setting_specs_cover_every_default_and_every_default_is_allowed():
    assert set(P.SETTING_SPECS) == set(C.DEFAULTS) == set(ev.Settings.__dataclass_fields__)
    for k, spec in P.SETTING_SPECS.items():
        assert spec.default == C.DEFAULTS[k]
        if spec.options is not None:
            assert spec.default in spec.options
        else:
            assert spec.lo <= spec.default <= spec.hi


def test_url_parameter_names_are_unique():
    names = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(names) == len(set(names))


def test_the_presets_name_the_settings_of_their_experiment():
    assert C.PRESETS["d = 10"]["d1"] == 10 and C.PRESETS["Gerade in R^10"]["art3"] == C.ART3[0] and C.PRESETS["Zwei Atome"]["art3"] == C.ART3[2]
    assert C.PRESETS["Minibatch d = 10"]["modus2"] == C.MODUS2[1] and C.PRESETS["Minibatch d = 10"]["dm"] == 10


@pytest.mark.parametrize("spec_key,roh,erwartet", [
    ("d1", 7, 5),                    # rastet auf die nächste Option ein
    ("d1", "10", 10),
    ("nmax1", 100000, 1024),
    ("wdh1", 40, 30),
    ("sigma3", 0.12, 0.1),
    ("sigma3", 0.7, 1.0),
    ("seed_input", 5_000_000, C.SEED_MAX),
    ("seed_input", -3, 0),
    ("modus2", "Minibatch (n = k)", "Minibatch (n = k)"),
    ("modus2", "irgendwas", None),   # ungültige Zeichenfolge wird ignoriert
])
def test_permalink_wert_wird_begrenzt_eingerastet_oder_ignoriert(spec_key, roh, erwartet):
    spec = P.SETTING_SPECS[spec_key]
    try:
        wert = P._eingerastet(spec, spec.caster(roh))
    except (ValueError, TypeError):
        wert = None
    assert wert == erwartet


def test_permalink_nan_wird_ignoriert():
    assert P._eingerastet(P.SETTING_SPECS["sigma3"], float("nan")) is None
