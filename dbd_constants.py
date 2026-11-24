"""Konstanten: Regler-Auswahlen, Standardwerte, Presets und Farben."""

SEED_MAX, DEFAULT_SEED = 999999, 2026

# Experiment 1: Steigung gegen Dimension
D1 = (1, 2, 3, 4, 5, 10)
NMAX1 = (256, 512, 1024)
WDH1 = (10, 30, 100)

# Experiment 2: Rauschboden und Minibatch
MODUS2 = ("Zwei Stichproben", "Minibatch (n = k)")
N2 = (32, 64, 128, 256, 512, 1024)
D2 = D1
WDH2 = (10, 30, 100)
DM = (2, 10)
FORMM = ("verschoben und gestreckt", "gleiche Verteilung (P = Q)")
SEEDSM = (200, 500, 2000)
KM = (8, 16, 32, 64, 128)

# Experiment 3: Struktur (R^10)
ART3 = ("Gerade", "Gerade + Rauschen", "Zwei Atome", "Zwei Blöcke")
SIGMA3 = (0.1, 0.3, 1.0)
WDH3 = (10, 30, 100)
D3 = 10

DEFAULTS = {
    "seed_input": DEFAULT_SEED,
    "d1": 1, "nmax1": 1024, "wdh1": 30,
    "modus2": MODUS2[0], "n2": 256, "d2": 1, "wdh2": 100,
    "dm": 10, "formm": FORMM[0], "seedsm": 500, "km": 128,
    "art3": ART3[1], "sigma3": 0.1, "wdh3": 30,
}

# Ein Preset legt nur die Regler fest, die es gegenüber den Standardwerten ändert; alle anderen Regler gehen auf ihren Standardwert zurück.
PRESETS = {
    "d = 1": {},
    "d = 10": {"d1": 10, "d2": 10},
    "Gerade in R^10": {"art3": ART3[0]},
    "Zwei Atome": {"art3": ART3[2]},
    "Minibatch d = 10": {"modus2": MODUS2[1], "dm": 10},
}
PRESET_HELP = {
    "d = 1": "Eine Dimension, Standardnormal: Der Zwei-Stichproben-Abstand fällt mit n etwa wie n hoch minus 1/2 (Steigung um −0,5, nicht −1).",
    "d = 10": "Zehn Dimensionen, Standardnormal: Der Abstand zweier Stichproben aus derselben Verteilung sinkt zwischen n = 32 und 1 024 nur mit der Steigung um −0,11.",
    "Gerade in R^10": "Zehn Dimensionen, aber alle Punkte liegen auf einer Geraden: das Problem ist eindimensional, die Steigung liegt wie bei d = 1 um −0,5 (bei 30 Wiederholungen mit einer Streuung von etwa ±0,1 von Satz zu Satz).",
    "Zwei Atome": "Zwei Punkte im Abstand 6, jede Stichprobe verteilt die Masse zufällig auf beide: die Steigung ist in jeder Dimension −1/2 (exakt −0,499 bei n = 32 bis 1 024).",
    "Minibatch d = 10": "Experiment 2 im Teil Minibatch: Zuordnung zwischen je k zufälligen Punkten in zehn Dimensionen, gemittelt über die Zufalls-Seeds.",
}

# Farben
COLOR_MAIN = "#2F6B65"
COLOR_FIT = "#d62728"
COLOR_REF = "rgba(120,120,120,0.55)"
COLOR_TRUTH = "#ff7f0e"
COLOR_LINE = "#4c78a8"
