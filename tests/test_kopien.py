"""Kopier-Tests: wörtlich kopierte Module sind gleich ihrer Quelle (eine Korrektur an einer Stelle darf in der anderen nicht fehlen).

Die Quellen liegen neben dem Demo-Ordner (Projects-Ordner); fehlen sie (CI, anderer Rechner), überspringt sich der Test."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUELLEN = {
    "dbd_rng.py": ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py",
    "dbd_zuordnung.py": ROOT.parent / "zuordnen-transportieren-demo" / "ztr_zuordnung.py",
}


def _lesen(pfad: Path) -> bytes:
    return pfad.read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("kopie", sorted(QUELLEN))
def test_kopie_ist_gleich_der_quelle(kopie):
    quelle = QUELLEN[kopie]
    if not quelle.exists():
        pytest.skip(f"Quelle {quelle} nicht vorhanden")
    assert _lesen(ROOT / kopie) == _lesen(quelle)


@pytest.mark.parametrize("kopie", sorted(QUELLEN))
def test_kopie_ist_nicht_leer_und_importierbar(kopie):
    text = (ROOT / kopie).read_text(encoding="utf-8")
    assert len(text) > 500
    __import__(kopie[:-3])
