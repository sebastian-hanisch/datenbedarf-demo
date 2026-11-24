"""Setzt die README aus `tools/readme_vorlage.md` zusammen: die Platzhalter {{name}} werden durch die Tabellen und Sätze aus `tests/readme_zeilen.py` ersetzt ({{testzahl}} durch die Zahl
der gesammelten Tests).

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/readme_bauen.py
Erst `tools/messreihen.py` (die Messdatei), dann dieses Skript, dann `tests/test_claims.py`. Die README ist ein erzeugtes Dokument: Änderungen an Text gehören in die Vorlage."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import readme_zeilen as R  # noqa: E402


def testzahl() -> int:
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=600)
    return int(re.search(r"(\d+) tests? collected", r.stdout).group(1))


def main():
    text = (ROOT / "tools" / "readme_vorlage.md").read_text(encoding="utf-8")
    for name, f in R.STUECKE.items():
        text = text.replace("{{" + name + "}}", f())
    if "{{testzahl}}" in text:
        text = text.replace("{{testzahl}}", str(testzahl()))
    offen = re.findall(r"\{\{\w+\}\}", text)
    if offen:
        raise SystemExit(f"nicht ersetzte Platzhalter: {offen}")
    (ROOT / "README.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"README.md geschrieben ({len(text)} Zeichen)")


if __name__ == "__main__":
    main()
