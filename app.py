"""Wie viele Daten braucht ein Abstand? Dimension, Struktur, Minibatch - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Der Wasserstein-Abstand zweier Stichproben aus derselben Verteilung ist nicht null: Rauschboden, Steigung gegen die Dimension, Struktur der Daten und Minibatch-Zuordnung,
gemessen im Standardnormal mit exakter Zuordnung gleicher Gewichte. Drei Experimente und ein Abschnitt zum geschnittenen Wasserstein-Abstand (sliced).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import dbd_constants as C
import dbd_evaluation as E
import dbd_vorgerechnet as VG
from dbd_layout import dunkles_design, kopf_css
from dbd_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from dbd_visualization import build_exp1, build_exp2_wahrheit, build_exp3, build_minibatch, build_zwei_punkt

st.set_page_config(page_title="Wie viele Daten braucht ein Abstand? Dimension, Struktur, Minibatch – Sebastian Hanisch", layout="wide")
KONFIG = {"displayModeBar": False}


@st.cache_data(show_spinner="Die Zuordnungen für Experiment 1 werden gerechnet ...")
def _analyse1(seed, d1, nmax1, wdh1):
    return E.analyse1(E.Settings(seed_input=seed, d1=d1, nmax1=nmax1, wdh1=wdh1))


@st.cache_data(show_spinner="Die Zwei-Stichproben-Abstände werden gerechnet ...")
def _analyse2_zwei(seed, n2, d2, wdh2):
    return E.analyse2_zwei(E.Settings(seed_input=seed, n2=n2, d2=d2, wdh2=wdh2))


@st.cache_data(show_spinner="Die Minibatch-Zuordnungen werden gerechnet (ein Batch je Zufalls-Seed) ...")
def _analyse2_mb(seed, dm, formm, seedsm):
    return E.analyse2_mb(E.Settings(seed_input=seed, dm=dm, formm=formm, seedsm=seedsm))


@st.cache_data(show_spinner="Die Zuordnungen für Experiment 3 werden gerechnet ...")
def _analyse3(seed, art3, sigma3, wdh3):
    return E.analyse3(E.Settings(seed_input=seed, art3=art3, sigma3=sigma3, wdh3=wdh3))


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in dbd_layout.py).
st.html(kopf_css(dunkles_design()))
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')
with st.container(key="kopf"):
    st.title("📉 Wie viele Daten braucht ein Abstand? Dimension, Struktur, Minibatch")
st.markdown(
    r"""
Zwei Stichproben aus **derselben** Verteilung haben nicht den Abstand null: der **Zwei-Stichproben-Abstand** (englisch two-sample distance), der Wasserstein-Abstand der zwei empirischen Verteilungen, ist eine Zufallsgröße,
und sein Wert bei gleicher Verteilung heißt **Rauschboden** (englisch noise floor). Die Demo misst im Standardnormal $N(0, I_d)$ mit exakter Zuordnung gleicher Gewichte, wie der Rauschboden mit der Zahl $n$ der Punkte
sinkt und wie das von der **Dimension** $d$ abhängt (**Fluch der Dimension**, englisch curse of dimensionality), und prüft die Annahme „der Abstand der Stichproben ist der Abstand der Verteilungen“: In einer Dimension gilt
sie bis auf den Faktor $\sqrt{2}$, in zehn Dimensionen bei $n = 1\,024$ nicht (der Abstand liegt dort noch bei rund 46 % des mittleren Paarabstands, obwohl der wahre Abstand null ist).

Weiter unten: **❓ Kernfragen**, die drei **🔬 Experimente** (Steigung gegen Dimension, Rauschboden und Minibatch, Struktur), der Abschnitt **🔀 Geschnittener Wasserstein-Abstand**, **🚧 Wo die Annahmen enden** und die Expander
**⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**. Gezählt wird in **Wiederholungen** (Zufalls-Seeds): jede Wiederholung zieht zwei neue Stichproben.
"""
)

with st.expander("So funktioniert der Abstand zweier Stichproben", expanded=True):
    st.markdown(
        r"""
1. **Eine Zuordnung, ein Abstand.** Zwei Stichproben $X_1, \dots, X_n$ und $Y_1, \dots, Y_n$ sind unabhängige Ziehungen aus $N(0, I_d)$. Jeder Punkt der einen wird genau einem der anderen zugeordnet (englisch assignment); **W1** ist die kleinste
   mittlere euklidische Entfernung der Paare, **W2** die Wurzel der kleinsten mittleren quadrierten Entfernung. In einer Dimension ist die sortierte Zuordnung die beste, sonst rechnet die Demo die Zuordnung exakt (`scipy.optimize.linear_sum_assignment`, bei fehlendem scipy mit einer eigenen Ungarischen Methode).
2. **Rauschboden und Maßstab.** Der wahre Abstand ist null, der gemessene nicht. Als Maßstab dient der mittlere Paarabstand zweier unabhängiger Punkte, $\mathbb{E}\lVert X - X'\rVert = 2\,\Gamma\!\left(\tfrac{d+1}{2}\right)/\Gamma\!\left(\tfrac{d}{2}\right)$; angegeben wird W1 durch diesen Maßstab.
3. **Steigung.** Die Gerade durch die Mittelwerte in log-log (Kleinste Quadrate) heißt **Steigung** (englisch slope) und sagt, wie schnell der Abstand mit $n$ sinkt. Nach Fournier und Guillin (Satz 1, gelesen) ist die obere Schranke für W1 bei genug Momenten $n^{-1/2}$ bei $d = 1$, $n^{-1/2}\log(1+n)$ bei $d = 2$ und $n^{-1/d}$ ab $d = 3$. Die Schranke gilt für den Abstand einer Stichprobe zur Verteilung; für den Abstand zweier Stichproben aus derselben Verteilung folgt dieselbe obere Schranke über die Dreiecksungleichung.
   Die Messung liegt bei $n = 32$ bis $1\,024$ und ist deshalb vorasymptotisch: die gemessene Steigung weicht von der Schranke ab (bei $d = 10$ −0,11 gegen −0,1).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben

    st.header("🔬 Experiment 1: Steigung")
    d1 = st.select_slider("Dimension d", C.D1, key="d1", help="Dimension des Standardnormals N(0, I_d).")
    nmax1 = st.select_slider("Größtes n", C.NMAX1, key="nmax1", format_func=lambda n: f"{n:,}".replace(",", " "), help="Das n-Raster ist 32, 64, ... bis zu diesem Wert (Verdopplungen); je größer, desto länger die Rechnung.")
    wdh1 = st.select_slider("Wiederholungen", C.WDH1, key="wdh1", help="Zahl der Wiederholungen je n (jede zieht zwei neue Stichproben). " + E.streuung_text())

    st.header("🔬 Experiment 2: Rauschboden und Minibatch")
    modus2 = st.radio("Teil", C.MODUS2, key="modus2", help="Zwei Stichproben: der Abstand zweier Stichproben aus derselben Verteilung. Minibatch: das Zuordnungsproblem zwischen je k zufällig gezogenen Punkten.")
    zwei = modus2 == C.MODUS2[0]
    n2 = st.select_slider("n (Zwei Stichproben)", C.N2, key="n2", format_func=lambda n: f"{n:,}".replace(",", " "), help="Punkte je Stichprobe.", disabled=not zwei)
    d2 = st.select_slider("Dimension d (Zwei Stichproben)", C.D2, key="d2", help="Nur für den Teil Zwei Stichproben.", disabled=not zwei)
    wdh2 = st.select_slider("Wiederholungen (Zwei Stichproben)", C.WDH2, key="wdh2", help="Zahl der Wiederholungen (Zufalls-Seeds) für den Mittelwert und die d = 1-Tabelle.", disabled=not zwei)
    dm = st.radio("Dimension d (Minibatch)", C.DM, key="dm", horizontal=True, help="Nur für den Teil Minibatch.", disabled=zwei)
    formm = st.radio("Verteilungen (Minibatch)", C.FORMM, key="formm", help="Verschoben und gestreckt: P = N(0, I) gegen Q = N(2 e1, 1,5² I), wahres W2² = 4 + d · 0,25. Gleiche Verteilung: P = Q, der wahre Abstand ist 0.", disabled=zwei)
    seedsm = st.select_slider("Zufalls-Seeds (Minibatch)", C.SEEDSM, key="seedsm", help="Zahl der Batches je k (jeder Batch ist ein Zufalls-Seed); je mehr, desto kleiner der Standardfehler.", disabled=zwei)
    km = st.select_slider("k (Minibatch, markiert)", C.KM, key="km", help="Die Kurve zeigt immer alle k; hier wird eines markiert und im Verdikt genannt.", disabled=zwei)

    st.header("🔬 Experiment 3: Struktur")
    art3 = st.radio("Daten in R^10", C.ART3, key="art3", help="Gerade: alle Punkte auf einer Achse. Gerade + Rauschen: dazu Streuung σ auf den neun anderen Achsen. Zwei Atome: je Stichprobe zwei Punkte im Abstand 6 (zufälliger Anteil). Zwei Blöcke: dasselbe mit Streuung σ je Achse.")
    sigma3 = st.select_slider("Streuung σ", C.SIGMA3, key="sigma3", format_func=lambda x: f"{x}".replace(".", ","), help="Nur für Gerade + Rauschen (σ auf den Nebenachsen; σ = 1 ist die volle Dimension) und Zwei Blöcke.", disabled=art3 in (C.ART3[0], C.ART3[2]))
    wdh3 = st.select_slider("Wiederholungen (Struktur)", C.WDH3, key="wdh3", help="Zahl der Wiederholungen je n.")

    st.header("🎲 Zufall")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für alle Stichproben der drei Experimente (die Bezugsreihen und die Tabelle des Abschnitts sliced sind mit Zufalls-Seed 2026 vorgerechnet).")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

werte = {k: st.session_state[k] for k in C.DEFAULTS}
sync_query_params(werte)
s = E.settings_aus_zustand(werte)

ref = E.referenz_reihen() or {}
mess = VG.lade()

if not E.W.scipy_verfuegbar():
    st.warning("scipy ist nicht installiert: die Zuordnungen rechnet die eigene Ungarische Methode (derselbe Wert, aber je nach Daten etwa 10- bis 45-mal, bei zwei Atomen über 100-mal langsamer, siehe README); n bis 256 ist empfohlen.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
if ref and mess.get("wahrheit") and mess.get("minibatch") and mess.get("struktur"):
    st.markdown("**Wie schnell sinkt der Abstand zweier Stichproben mit n, und wovon hängt es ab?**")
    st.info(
        f"Zwischen n = 32 und 1 024 fällt W1 mit der Steigung {E.zahl(ref['1']['steigung_alle'], 2)} / {E.zahl(ref['2']['steigung_alle'], 2)} / {E.zahl(ref['5']['steigung_alle'], 2)} / {E.zahl(ref['10']['steigung_alle'], 2)} für d = 1 / 2 / 5 / 10 "
        f"(Zufalls-Seed 2026; {ref['1']['wdh']} Wiederholungen bei d = 1, {ref['2']['wdh']} sonst). Bei d = 1 sind es −0,5, nicht −1. Bei d = 10 und n = 1 024 liegt der Abstand zweier Stichproben aus derselben Verteilung noch bei "
        f"{E.prozent(ref['10']['verhaeltnis'][-1])} des mittleren Paarabstands, bei d = 5 bei {E.prozent(ref['5']['verhaeltnis'][-1])}, bei d = 1 bei {E.prozent(ref['1']['verhaeltnis'][-1])} (Experiment 1)."
    )
    st.markdown("**Ist der Zwei-Stichproben-Abstand ein Maß für den Abstand der Verteilungen, und was kostet ein Minibatch?**")
    wz = mess["wahrheit"]["zeilen"]
    mb = mess["minibatch"]["reihen"]
    st.info(
        f"Bei d = 1 liegt er im Mittel etwa {E.zahl(sum(z['verhaeltnis'] for z in wz) / len(wz), 1)}-mal über dem Abstand zur Wahrheit (√2 im Grenzwert); für d ≥ 2 ist dieser Faktor nicht gemessen. Der Minibatch-Wert zwischen gleichen Verteilungen "
        f"ist derselbe Rauschboden mit n = k: bei d = 10 und k = 128 {E.zahl(mb['10_gleich']['zeilen'][-1]['mittel'], 2)} statt 0 (zufällige Paarung: 20; {E.ganz(mess['minibatch']['seeds'])} Zufalls-Seeds); zwischen verschiedenen Verteilungen liegt er im Mittel "
        f"nie unter dem wahren W2² (Experiment 2)."
    )
    st.markdown("**Hilft Struktur?**")
    st_ = mess["struktur"]
    st.info(
        f"In R^10 mit 100 Wiederholungen: Daten auf einer Geraden {E.zahl(st_['gerade_0.0']['steigung_alle_100'], 2)} (wie d = 1), mit Streuung σ = 0,1 auf den Nebenachsen {E.zahl(st_['gerade_0.1']['steigung_alle_100'], 2)}, mit σ = 0,3 {E.zahl(st_['gerade_0.3']['steigung_alle_100'], 2)}, "
        f"volle Dimension {E.zahl(st_['gerade_1.0']['steigung_alle_100'], 2)}. Zwei Atome haben als Erwartung exakt die Steigung {E.zahl(E.exakte_steigung_atome(), 3)} (n = 32 bis 1 024, in jeder Dimension). Struktur hilft in dem Maß, in dem sie exakt ist (Experiment 3)."
    )
else:
    st.warning("Die vorgerechneten Reihen (`tools/messreihen.json`) fehlen; die Experimente laufen trotzdem.")

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Steigung gegen Dimension, nötiges n")
st.caption(
    "W1 zwischen zwei Stichproben aus N(0, I_d) mit n Punkten, durch den mittleren Paarabstand geteilt; Mittel über die Wiederholungen (Fehlerbalken: Standardfehler), Zufalls-Seed und Wiederholungen wie in der Seitenleiste. "
    "Grau: die Bezugsreihen der übrigen Dimensionen (Zufalls-Seed 2026, 200 Wiederholungen bei d = 1, sonst 30). Der wahre Abstand ist überall null."
)
a1 = _analyse1(s.seed_input, s.d1, s.nmax1, s.wdh1)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Steigung (alle n)", E.zahl(a1["steigung_alle"], 2), help="Kleinste Quadrate auf log W1 gegen log n über das n-Raster.")
m2.metric("Steigung (letzte drei n)", E.zahl(a1["steigung_letzte3"], 2) if len(a1["ns"]) >= 3 else "–")
m3.metric(f"W1 durch Maßstab bei n = {E.ganz(a1['ns'][-1])}", E.prozent(a1["verhaeltnis"][-1], 1), help=f"Maßstab: mittlerer Paarabstand {E.zahl(a1['massstab'], 3)}.")
m4.metric("Erwartung (obere Schranke)", E.erwartung_text(s.d1), help="Fournier/Guillin, Satz 1, p = 1: −1/2 bei d = 1, −1/2 mit Log-Faktor bei d = 2, −1/d ab d = 3 (für große n). Die Schranke gilt für den Abstand Stichprobe gegen Verteilung; für zwei Stichproben folgt sie über die Dreiecksungleichung.")
st.plotly_chart(build_exp1(a1, ref), width="stretch", key="exp1", config=KONFIG)
n10, st10 = a1["noetig"][0.10]
n5, st5 = a1["noetig"][0.05]
tab1 = []
for ds in sorted(ref, key=int) if ref else []:
    r_ = ref[ds]
    tab1.append([ds, E.zahl(r_["steigung_alle"], 2), E.zahl(r_["steigung_letzte3"], 2), E.prozent(r_["verhaeltnis"][0], 0), E.prozent(r_["verhaeltnis"][-1], 0), E.zahl(E.halbierungsfaktor(r_["steigung_alle"]), 0)])
if tab1:
    st.markdown("**Bezugsreihen je Dimension** (Zufalls-Seed 2026, n = 32 bis 1 024; Halbierungsfaktor: um diesen Faktor müsste n wachsen, damit der Abstand bei der gemessenen Steigung auf die Hälfte fällt, Rechnung):")
    st.markdown(E.tabelle_md(["d", "Steigung (alle n)", "Steigung (letzte drei n)", "W1 durch Maßstab bei n = 32", "bei n = 1 024", "Halbierungsfaktor"], tab1))
st.markdown(
    f"**Nötiges n** (Potenzanpassung der gewählten Reihe): für W1 unter 10 % des mittleren Paarabstands n ≈ {E.gross_n(n10)}{'*' if st10 else ''}, unter 5 % n ≈ {E.gross_n(n5)}{'*' if st5 else ''}. "
    "Ein Stern heißt Extrapolation der Anpassung über das gemessene n-Raster hinaus; liegt die Anpassung nur knapp darüber (bei d = 1: 1 040 gegen n = 1 024), kann der gemessene Mittelwert bei n = 1 024 schon unter dem Ziel liegen."
)
ex = VG.teil("extrapolation")
if ex:
    zeilen_ex = []
    for ds in sorted(ex, key=int):
        z0, z1 = ex[ds]["zeilen"]
        zeilen_ex.append([ds, str(ex[ds]["wdh"]), f"{E.zahl(z0['verhaeltnis'], 3)} (± {E.prozent(z0['se'] / z0['gemessen'], 1)})", f"{E.zahl(z1['verhaeltnis'], 3)} (± {E.prozent(z1['se'] / z1['gemessen'], 1)})"])
    st.markdown("**Extrapolation um den Faktor 4 geprüft** (Verhältnis gemessen zu vorhergesagt aus der Anpassung der Bezugsreihe; Klammer: Standardfehler des Messwerts):")
    st.markdown(E.tabelle_md(["d", "Wiederholungen", "n = 2 048", "n = 4 096"], zeilen_ex))
kind, text = E.verdikt1(s, a1)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Rauschboden und Minibatch – der Abstand zweier gleicher Verteilungen ist nicht null")
if s.modus2 == C.MODUS2[0]:
    st.caption(
        "Teil **Zwei Stichproben**: zwei Stichproben aus derselben Verteilung N(0, I_d), der wahre Abstand ist null. Bei d = 1 steht daneben der Abstand einer Stichprobe zur Wahrheit N(0, 1), "
        "∫ |F_n − Φ| (exakt per Quadratur). Der Teil Minibatch in der Seitenleiste rechnet dieselbe Größe mit n = k (Umschalter „Teil“)."
    )
    a2 = _analyse2_zwei(s.seed_input, s.n2, s.d2, s.wdh2)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"W1 bei d = {a2['d']}, n = {E.ganz(a2['n'])}", E.zahl(a2["w1_mittel"], 3), help=f"Mittel über {a2['wdh']} Wiederholungen, Standardfehler {E.zahl(a2['w1_se'], 3)}; wahrer Abstand: 0.")
    m2.metric("W1 durch mittleren Paarabstand", E.prozent(a2["anteil"], 1), help=f"Maßstab {E.zahl(a2['massstab'], 3)}.")
    if a2["wahrheit"]:
        m3.metric("Stichprobe gegen Wahrheit (d = 1)", E.zahl(a2["wahrheit"]["wahrheit"], 3), help=f"Verhältnis Zwei Stichproben zu Wahrheit: {E.zahl(a2['wahrheit']['verhaeltnis'], 2)}.")
    else:
        m3.metric("Stichprobe gegen Wahrheit", "nicht gemessen", help="Für d ≥ 2 gibt es keine einfache geschlossene Berechnung (die eindimensionale Struktur fehlt), die Demo rechnet den Abstand zur Wahrheit dort nicht; der Zwei-Stichproben-Abstand steht stellvertretend, das ist eine Voraussetzung.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_exp2_wahrheit(a2["tabelle_d1"]), width="stretch", key="exp2_wahrheit", config=KONFIG)
    c2.plotly_chart(build_zwei_punkt(), width="stretch", key="exp2_zwei_punkt", config=KONFIG)
    st.caption("Links: d = 1, alle n, Wiederholungen wie in der Seitenleiste. Rechts: Zwei-Punkt-Verteilung ½ δ₀ + ½ δ₁, exakt aus den Binomialformeln (keine Zufallszahlen).")
    tab2 = [[E.ganz(z["n"]), E.zahl(z["zwei"], 4), E.zahl(z["wahrheit"], 4), E.zahl(z["verhaeltnis"], 2)] for z in a2["tabelle_d1"]]
    st.markdown("**d = 1, Zwei-Stichproben-W1 gegen Abstand zur Wahrheit:**")
    st.markdown(E.tabelle_md(["n", "Zwei Stichproben", "Stichprobe gegen Wahrheit", "Verhältnis"], tab2))
    tab2p = [[str(z["N"]), E.zahl(z["zwei"], 5), E.zahl(z["wahr"], 5), E.zahl(z["verhaeltnis"], 4)] for z in a2["zwei_punkt"]]
    st.markdown(r"**Zwei-Punkt-Verteilung, exakt:** $\mathbb{E}\lvert Z - Z'\rvert = \binom{2N}{N} / 4^N$ (zwei Stichproben) und $\mathbb{E}\lvert Z - \tfrac12\rvert = \binom{N}{N/2} / 2^{N+1}$ (Wahrheit), $Z$ der Anteil der Einsen:")
    st.markdown(E.tabelle_md(["N", "Zwei Stichproben", "Wahrheit", "Verhältnis"], tab2p))
    kind, text = E.verdikt2_zwei(s, a2)
    getattr(st, kind)(text)
else:
    st.caption(
        "Teil **Minibatch**: Das Zuordnungsproblem zwischen je k zufällig gezogenen Punkten aus P und aus Q, gemittelt über die Zufalls-Seeds (die Teilstichproben-Zuordnung, englisch minibatch assignment). "
        "Wert = kleinste mittlere quadrierte Entfernung über alle Zuordnungen; V = Wert durch das wahre W2² zwischen P und Q. Bei P = Q ist das wahre W2² null: der Wert ist reiner Rauschboden, "
        "derselbe wie bei zwei Stichproben der Größe n = k."
    )
    a2 = _analyse2_mb(s.seed_input, s.dm, s.formm, s.seedsm)
    zk = E.zeile_k(a2, s.km)
    m1, m2, m3 = st.columns(3)
    if a2["form"] == "verschoben":
        m1.metric(f"V bei k = {s.km}", E.zahl(zk["v"], 3), help=f"Minibatch-Wert {E.zahl(zk['mittel'], 3)} durch wahres W2² {E.zahl(a2['wahr'], 2)}; Standardfehler {E.zahl(zk['v_se'], 3)}.")
        m2.metric("Batches mit V unter 1", E.prozent(zk["anteil_unter_1"], 0), help="Anteil der einzelnen Batches, deren Wert unter dem wahren W2² liegt; im Mittel liegt V nie darunter.")
        m3.metric("wahres W2² (P gegen Q)", E.zahl(a2["wahr"], 2))
    else:
        m1.metric(f"Minibatch-Wert bei k = {s.km}", E.zahl(zk["mittel"], 2), help=f"Standardfehler {E.zahl(zk['se'], 3)}; wahrer Abstand 0.")
        m2.metric("Zufällige Paarung", E.zahl(a2["zufallspaarung"], 0), help="Mittlere quadrierte Entfernung bei zufälliger Paarung: 2d.")
        m3.metric("Anteil an der Zufallspaarung", E.prozent(zk["mittel"] / a2["zufallspaarung"], 0))
    st.plotly_chart(build_minibatch(a2, s.km), width="stretch", key="exp2_minibatch", config=KONFIG)
    if a2["form"] == "verschoben":
        tabmb = [[str(z["k"]), f"{E.zahl(z['mittel'], 3)} ± {E.zahl(z['se'], 3)}", f"{E.zahl(z['v'], 3)} ± {E.zahl(z['v_se'], 3)}", E.prozent(z["anteil_unter_1"], 0)] for z in a2["zeilen"]]
        st.markdown(E.tabelle_md(["k", "Minibatch-Wert", "V", "Batches mit V unter 1"], tabmb))
        if a2["d"] == 2:
            sl_ = E.S.steigung_v_minus_1(a2["zeilen"])
            st.markdown(f"Steigung von log(V − 1) gegen log k zwischen k = 32 und 128: {E.zahl(sl_, 2)} (dieser Zufalls-Seed-Satz); sie streut von Satz zu Satz (siehe README), bei d = 10 ist sie stabil bei etwa −0,23.")
        else:
            sl_ = E.S.steigung_v_minus_1(a2["zeilen"])
            st.markdown(f"Steigung von log(V − 1) gegen log k zwischen k = 32 und 128: {E.zahl(sl_, 2)}.")
    else:
        zs = {z["k"]: z for z in a2["zwei_stichproben"]}
        tabmb = [[str(z["k"]), f"{E.zahl(z['mittel'], 3)} ± {E.zahl(z['se'], 3)}", f"{E.zahl(zs[z['k']]['mittel'], 3)} ± {E.zahl(zs[z['k']]['se'], 3)}"] for z in a2["zeilen"]]
        st.markdown(E.tabelle_md(["k", "Minibatch-Wert bei P = Q", "Zwei Stichproben, W2² mit n = k"], tabmb))
    kind, text = E.verdikt2_mb(s, a2)
    getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Struktur hilft in dem Maß, in dem sie exakt ist")
st.caption(
    "Zwei-Stichproben-W1 in R^10 für Daten mit Struktur: alle Punkte auf einer Achse (Gerade), dazu Streuung auf den neun anderen Achsen, zwei Atome oder zwei Blöcke bei −3 e₁ und +3 e₁ (Abstand 6). "
    "Grau: volle Dimension (Standardnormal in R^10, Bezugsreihe mit Zufalls-Seed 2026 und 30 Wiederholungen). Absolute Werte, nicht durch den Maßstab geteilt."
)
a3 = _analyse3(s.seed_input, s.art3, s.sigma3, s.wdh3)
m1, m2, m3 = st.columns(3)
m1.metric("Steigung (alle n)", E.zahl(a3["steigung_alle"], 2), help="Kleinste Quadrate auf log W1 gegen log n, n = 32 bis 1 024.")
m2.metric("Steigung (letzte drei n)", E.zahl(a3["steigung_letzte3"], 2))
m3.metric("Volle Dimension (Bezug)", E.zahl(a3["volle_dimension"]["steigung_alle"], 2) if a3["volle_dimension"] else "–", help="Standardnormal in R^10, Zufalls-Seed 2026, 30 Wiederholungen.")
st.plotly_chart(build_exp3(a3), width="stretch", key="exp3", config=KONFIG)
if "exakt" in a3:
    st.markdown(
        r"**Zwei Atome, exakt:** bei Abstand $\Delta = 6$ ist $W_1 = \Delta\,\lvert Z - Z'\rvert$ ($Z$, $Z'$ die Anteile der Punkte am rechten Atom), also $\mathbb{E}\,W_1 = 6 \binom{2n}{n}/4^n$: "
        f"{E.zahl(6 * float(E.W.zwei_punkt_zwei_stichproben(32)), 4)} bei n = 32 und {E.zahl(6 * float(E.W.zwei_punkt_zwei_stichproben(1024)), 4)} bei n = 1 024, Steigung {E.zahl(a3['exakt'], 3)}; die Dimension geht nicht ein."
    )
if "gerade_gegen_1d" in a3:
    st.markdown(f"Kontrolle: bei Punkten auf einer Geraden stimmt das W1 der allgemeinen Zuordnung in R^10 mit der sortierten eindimensionalen Zuordnung überein (Unterschied bei n = 128, erste Wiederholung: {E.exp_text(a3['gerade_gegen_1d'])}).")
kind, text = E.verdikt3(s, a3)
getattr(st, kind)(text)

st.markdown("---")

# --- Geschnittener Wasserstein-Abstand -----------------------------------------------------------------------------------------------------

with st.expander("🔀 Geschnittener Wasserstein-Abstand (sliced): ordnet, misst aber etwas anderes"):
    sl = E.tabelle_sliced()
    ver = E.SL.verhaeltnisse()
    st.markdown(
        r"""
Der **geschnittene Wasserstein-Abstand** (englisch sliced Wasserstein distance, hier „sliced“) mittelt die eindimensionalen Wasserstein-Abstände der Projektionen auf zufällige Richtungen; jede Projektion braucht nur das Sortieren.
Verglichen wird er mit dem vollen $W_2$ zwischen Gauß-Verteilungen in $\mathbb{R}^{10}$: Referenz $N(0, I_{10})$, 40 Vergleichsverteilungen $N(\mu, \Sigma)$ in vier Familien zu je zehn Stärken $m = 0{,}1; 0{,}2; \dots; 1{,}0$:
**S** Verschiebung um $m$ in einer Richtung; **I** isotrope Streuung, Standardabweichung $1 + m$ in allen Achsen; **E** Streuung nur in der ersten Achse, Standardabweichung $1 + 3m$; **K** Gleichkorrelation, alle Kovarianzen $\rho = m/2$.
Die Wahrheit ist $W_2$ geschlossen (Bures).
        """
    )
    st.markdown("**Verhältnis sliced zu W2 (Population, geschlossen) je Familie und Stärke:**")
    kopf = ["Familie"] + [E.zahl(m, 1) for m in E.SL.STAERKEN] + ["Mittel"]
    zl = [[f"{f} {E.SL.FAMILIEN_NAMEN[f]}"] + [E.zahl(v, 3) for v in ver[f]] + [E.zahl(sum(ver[f]) / len(ver[f]), 3)] for f in E.SL.FAMILIEN]
    st.markdown(E.tabelle_md(kopf, zl))
    st.caption("Das Verhältnis hängt an der Stärke und an ihrer Definition: bei anderen Skalen der Stärke ergeben sich andere Mittel. Bei Verschiebung und isotroper Streuung ist es bei jeder Stärke gleich 1/√d = 0,316.")
    if sl:
        st.markdown(f"**Rangübereinstimmung mit der Wahrheit** (Stichproben mit n = {sl['n']} Punkten, 40 Fälle, Zufalls-Seed {sl['seed']}; Spearman und Kendall τ):")
        zl2 = [
            ["voller Abstand W2 aus Stichproben (eine Stichprobe je Fall, ohne Streuungsangabe)",E.zahl(sl["voll_spearman"], 3), E.zahl(sl["voll_kendall"], 3), E.zahl(sl["voll_spearman_familie"], 3)],
            ["sliced, Population (geschlossen)", E.zahl(sl["pop_spearman"], 3), E.zahl(sl["pop_kendall"], 3), E.zahl(sl["pop_spearman_familie"], 3)],
        ] + [[f"sliced aus Stichproben, L = {z['L']} Richtungen ({sl['wdh']} Richtungssätze)", E.zahl(z["spearman"], 3), E.zahl(z["kendall"], 3), E.zahl(z["spearman_familie"], 3)] for z in sl["zeilen"]]
        st.markdown(E.tabelle_md(["Abstand", "Spearman", "Kendall τ", "Spearman je Familie (Mittel)"], zl2))
        st.markdown(
            f"Rauschboden des vollen $W_2$ zwischen zwei Stichproben aus $N(0, I_{{10}})$ mit n = {sl['n']} (Mittel über 5 Paare): {E.zahl(sl['rauschboden_w2'], 2)}; die Wahrheit liegt zwischen {E.zahl(sl['bures_min'], 2)} und {E.zahl(sl['bures_max'], 2)}."
        )
    st.info(
        "**Befund:** sliced ordnet für die Population innerhalb jeder Familie perfekt (Spearman 1,000), aus Stichproben mit n = 300 und L = 500 Richtungen mit Spearman 0,980 je Familie, über alle vier Familien nur mit Spearman 0,93, weil es Streuung in einer Achse und Korrelation schwächer gewichtet als eine Verschiebung: "
        "es misst eine andere, schwächere Distanz, nicht denselben Abstand schneller. Die Rate von sliced selbst ist hier nicht gemessen; nach der Literatur (Chewi, Niles-Weed und Rigollet; hier im Volltext nicht nachgeprüft) gilt für Maße im Einheitsball $\\mathbb{E}\\,SW_1 \\lesssim n^{-1/2}$."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme der Vorgänger | Was hier passiert, wenn sie nicht gilt | Wo es weitergeht |
|---|---|---|
| *Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1.* | Gegeben sind zwei **Stichproben**: der berechnete Abstand ist eine Zufallsgröße, und bei gleicher Verteilung ist er nicht null (Rauschboden, Experiment 2); wie schnell er mit n sinkt, hängt an der Dimension (Experiment 1) und an der Struktur der Daten (Experiment 3). Die Demo „Abstand zwischen Verteilungen“ zeigt Stichprobenrauschen nur an einem Beispiel (eine Dimension, ein wählbares n) und hebt die Annahme nicht auf. | Erkennen einer Verteilungsänderung mit Schwelle (spätere Demo der Reihe). |
| *Der Abstand der Stichproben ist der Abstand der Verteilungen.* | Gilt bei d = 1 bis auf den Faktor √2 ≈ 1,41, bei d = 10 und n = 1 024 nicht (46 % des mittleren Paarabstands bei wahrem Abstand null). Für d ≥ 2 steht der Zwei-Stichproben-Abstand stellvertretend für die Wahrheit, das ist eine Voraussetzung, nicht gemessen. | – |
| *Der Minibatch-Wert ist ein Näherungswert für den vollen Abstand.* | Zwischen gleichen Verteilungen ist er derselbe Rauschboden wie bei zwei Stichproben mit n = k (nicht null); zwischen verschiedenen liegt er im Mittel nie unter dem wahren W2². Eine isotrope Gauß-Familie, exakte Zuordnung, nur diese gemessen. | Generative Modelle (spätere Linie der Reihe). |
"""
)
st.markdown(
    "Zwei verschiedene Gründe, warum der Abstand einer Verteilung zu sich selbst nicht null ist: bei der entropischen Sinkhorn-Größe die Entropie (Demo „Abstand zwischen Verteilungen“), hier die Stichprobe: "
    "zwei Stichproben derselben Verteilung sind zwei verschiedene Verteilungen."
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Standardnormal $N(0, I_d)$ (und die Strukturen in R^10 von Experiment 3, die Gauß-Familien des Abschnitts sliced); **exakte Zuordnung mit gleichen Gewichten** $1/n$, W1 mit euklidischen Entfernungen.
- Zähleinheit jeder Mittelung ist die **Wiederholung** (Zufalls-Seed): zwei neue Stichproben je Wiederholung; im Minibatch ein Batch je Zufalls-Seed.
- Zufall aus dem Hausgenerator SplitMix64; die Normalzahlen sind Box-Muller mit numpy-Funktionen (in der letzten Binärstelle plattformabhängig, ohne Einfluss auf die gezeigten Stellen).
- Abstand zur Wahrheit nur bei d = 1 (dort exakt per Quadratur); für d ≥ 2 steht der Zwei-Stichproben-Abstand stellvertretend.

**Grenzen.**
- Die Messung liegt im Bereich n = 32 bis 1 024 und ist **vorasymptotisch**: die Steigungen sind keine Raten für n → ∞ (bei d = 10 gemessen −0,11 gegen −0,1 der Schranke). Die Formel $n^{-1/d}$ gilt als Schranke erst ab d ≥ 3.
- Steigungen sind keine genauen Werte: @@STREUUNG@@
- „Nötiges n“ ist eine Potenzanpassung; der Stern kennzeichnet Extrapolation. Um den Faktor 4 (n = 2 048 und 4 096) ist sie geprüft, darüber hinaus sind es Rechenbeispiele, keine Messung.
- Nur Standardnormal: Gleichverteilung im Würfel, Mischungen, schwere Ränder und Verteilungen mit P ≠ Q beim Zwei-Stichproben-Abstand sind nicht gemessen. Bei schweren Rändern gilt nach Fournier und Guillin eine andere Untergrenze.
- Minibatch: eine isotrope Gauß-Familie; anisotrop, Mischungen, mit Entropie und stetig gegen stetig sind nicht gemessen. Die Aussage „Verzerrung nicht wachsend in k, O(1/k) bei kompaktem Träger“ (Boïté, Delon, Nadjahi 2026, Preprint; nur die Zusammenfassung gelesen, die genaue Form steht im Volltext, der nicht geprüft ist) ist hier nicht geprüft; die Messung ist Gauß, nicht kompakt.
- Entropische Schätzer (Sinkhorn-Divergenz) mit ε-abhängigen Konstanten sind nicht gemessen. Zu Zwei-Stichproben-Tests mit W1 in hoher Dimension macht die Demo keine Aussage (die Voraussetzungen der Quelle, Klasse der Dichten und Art des Tests, sind hier nicht geprüft).
- Die Zuordnung rechnet `scipy.optimize.linear_sum_assignment`; bei fehlendem scipy die eigene Ungarische Methode (derselbe Wert, deutlich langsamer; Obergrenze in der README).

**Bekannte Probleme und Unsicherheiten.**
- Die Zahl der Wiederholungen und der Zufalls-Seed ändern die gezeigten Steigungen und Anteile (Streuung siehe oben); bei d = 2 streut V − 1 im Minibatch von Satz zu Satz von Zufalls-Seeds so stark, dass die Steigung von log(V − 1) nur als Band gilt.
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).
        """.replace("@@STREUUNG@@", E.streuung_text())
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* zwei Stichproben $X_1, \dots, X_n$ und $Y_1, \dots, Y_n$ aus derselben Verteilung $P$ auf $\mathbb{R}^d$. *Gesucht:* ihr Wasserstein-Abstand (mit gleichen Gewichten $1/n$) und wie er mit $n$ und $d$ sinkt, obwohl der wahre Abstand null ist.

| Symbol | Bedeutung |
|---|---|
| $n$, $d$ | Punkte je Stichprobe, Dimension |
| $X_i$, $Y_i$ | Punkte der beiden Stichproben (Zeilen der Matrizen $X$, $Y$) |
| $\pi$ | Zuordnung (Permutation) der Punkte $X_i \mapsto Y_{\pi(i)}$ |
| $W_1$, $W_2$ | Wasserstein-Abstände mit euklidischen bzw. quadrierten Entfernungen |
| Maßstab | mittlerer Paarabstand $\mathbb{E}\lVert X - X'\rVert$ zweier unabhängiger Punkte aus $P$ |
| Steigung $s$ | Steigung von $\log W_1$ gegen $\log n$ (Kleinste Quadrate) |
| $k$, $V$ | Punkte je Batch, Minibatch-Wert durch wahres $W_2^2$ |
| $\Delta$, $Z$ | Abstand der zwei Atome, Anteil der Punkte am rechten Atom |

**Zwei-Stichproben-Abstand.**
$$W_1(\hat P_n, \hat Q_n) = \min_{\pi}\ \frac1n \sum_{i=1}^{n} \lVert X_i - Y_{\pi(i)}\rVert, \qquad W_2^2(\hat P_n, \hat Q_n) = \min_{\pi}\ \frac1n \sum_{i=1}^{n} \lVert X_i - Y_{\pi(i)}\rVert^2$$
In einer Dimension ist die sortierte Zuordnung die beste: $W_1 = \frac1n \sum_i \lvert x_{(i)} - y_{(i)}\rvert$.

**Handbeispiel** ($d = 1$, $x = (0, 1, 4)$, $y = (0, 2, 2)$): sortiert gepaart $0 \leftrightarrow 0$, $1 \leftrightarrow 2$, $4 \leftrightarrow 2$, also
$$W_1 = \frac{0 + 1 + 2}{3} = 1, \qquad W_2^2 = \frac{0 + 1 + 4}{3} = \frac53 .$$

**Maßstab.** Für $X, X'$ unabhängig aus $N(0, I_d)$ ist $X - X' \sim N(0, 2 I_d)$, also
$$\mathbb{E}\lVert X - X'\rVert = 2\,\frac{\Gamma\!\left(\frac{d+1}{2}\right)}{\Gamma\!\left(\frac{d}{2}\right)} .$$

**Raten** (Fournier und Guillin 2015, Satz 1 mit $p = 1$ und genug Momenten, obere Schranken):
$$\mathbb{E}\,W_1(\hat P_n, P) \lesssim \begin{cases} n^{-1/2} & d = 1 \\ n^{-1/2}\log(1+n) & d = 2 \\ n^{-1/d} & d \ge 3 \end{cases}$$
Ein Maß mit zwei getrennten Trägerteilen hat in jeder Dimension mindestens $n^{-1/2}$ (Untergrenze, keine bessere Rate). Für $W_2$ gilt $n^{-1/d}$ erst ab $d > 4$; $W_2^2$ hat die doppelte Steigung von $W_2$. Die Schranken gelten für den Abstand einer Stichprobe zur Verteilung; für zwei Stichproben aus derselben Verteilung folgen sie über die Dreiecksungleichung $W_1(\hat P_n, \hat Q_n) \le W_1(\hat P_n, P) + W_1(P, \hat Q_n)$.

**Abstand zur Wahrheit bei $d = 1$.**
$$W_1(\hat P_n, N(0,1)) = \int_{-\infty}^{\infty} \lvert F_n(x) - \Phi(x)\rvert\,\mathrm{d}x$$
exakt stückweise: zwischen zwei Beobachtungen ist $F_n$ konstant, das Integral ist mit der Stammfunktion $x\Phi(x) + \varphi(x)$ von $\Phi$ geschlossen.

**Zwei-Punkt-Verteilung** $P = \tfrac12 \delta_0 + \tfrac12 \delta_1$, $Z$ und $Z'$ die Anteile der Einsen in zwei unabhängigen Stichproben der Größe $N$:
$$\mathbb{E}\lvert Z - Z'\rvert = \frac{\binom{2N}{N}}{4^N}, \qquad \mathbb{E}\lvert Z - \tfrac12\rvert = \frac{\binom{N}{N/2}}{2^{N+1}}$$
Beispiele: $N = 2$: $\tfrac38$ und $\tfrac14$ (Verhältnis $\tfrac32$); $N = 4$: $\tfrac{35}{128}$ und $\tfrac{3}{16}$. Das Verhältnis fällt gegen $\sqrt{2}$ (beide Abstände sind im Grenzwert gaußsche Prozesse, der Zwei-Stichproben-Prozess hat die doppelte Varianz), und $\sqrt{N}\,\mathbb{E}\lvert Z - Z'\rvert \to 1/\sqrt{\pi}$.
Zwei Atome im Abstand $\Delta$ in $\mathbb{R}^d$: $W_1 = \Delta\,\lvert Z - Z'\rvert$, die Dimension geht nicht ein, die Steigung ist $-\tfrac12$.

**Minibatch-Wert.** Mit je $k$ zufälligen Punkten $X_1, \dots, X_k \sim P$ und $Y_1, \dots, Y_k \sim Q$:
$$\mathrm{Wert}_k = \mathbb{E}\,\min_{\pi}\ \frac1k \sum_{i=1}^{k} \lVert X_i - Y_{\pi(i)}\rVert^2, \qquad V = \frac{\mathrm{Wert}_k}{W_2^2(P, Q)}$$
Bei $P = Q$ ist $W_2^2 = 0$ und $\mathrm{Wert}_k$ ist der Zwei-Stichproben-Abstand zum Quadrat mit $n = k$. Der Mittelwert der Batch-Pläne ist ein Plan zwischen $P$ und $Q$ und $W_2^2$ ist gemeinsam konvex, also
$$\mathbb{E}\,W_2^2(\hat P_k, \hat Q_k) \ \ge\ W_2^2(P, Q)$$
(im Mittel nie zu klein; einzelne Batches können darunter liegen). Zwei-Punkt-Beispiel $P = \tfrac12 \delta_0 + \tfrac12 \delta_1$, $Q = P + 2$ ($W_2^2 = 4$): $V = \tfrac98$ bei $k = 1$ und $\tfrac{35}{32}$ bei $k = 2$. Für $P = N(0, I_d)$ und $Q = N(2e_1, 1{,}5^2 I_d)$ ist $W_2^2 = 4 + 0{,}25\,d$.

**Geschnittener Abstand (sliced) und Gauß-Fall.** Mit gleichverteilten Richtungen $\theta$ auf der Einheitskugel:
$$SW_2^2(P, Q) = \mathbb{E}_\theta\, W_2^2(\theta_\# P, \theta_\# Q)$$
Für $N(0, I)$ gegen $N(\mu, \Sigma)$ ist das eindimensionale $W_2^2 = (\theta\cdot\mu)^2 + (\sqrt{\theta^\top \Sigma\,\theta} - 1)^2$, und
$$W_2^2 = \lVert\mu\rVert^2 + \sum_i \big(1 - \sqrt{\lambda_i}\big)^2 \quad (\lambda_i \text{ die Eigenwerte von } \Sigma).$$
Bei Verschiebung ist $\mathbb{E}\,(\theta\cdot\mu)^2 = \lVert\mu\rVert^2/d$, also $SW_2 / W_2 = 1/\sqrt{d}$ (0,316 bei $d = 10$).

**Literatur** (Was nicht im Volltext geprüft ist, ist so gekennzeichnet):Fournier, Guillin (2015), *On the rate of convergence in Wasserstein distance of the empirical measure*, Probab. Theory Related Fields 162, 707–738, arXiv 1312.2128 (Satz 1, Abschn. 1.2 gelesen);
Chewi, Niles-Weed, Rigollet, *Statistical optimal transport* (Saint-Flour-Skript 2024), arXiv 2407.18163 (Titel und Autoren geprüft; die Aussage zu sliced und ihr Kapitel sind im Volltext nicht nachgeprüft); Kuhn, Shafiee, Wiesemann (2025), *Distributionally robust optimization*, Acta Numerica 34, arXiv 2411.02549 (Satz 10.2: der Radius der Wasserstein-Kugel sinkt wie $N^{-1/d}$; Satznummer und Wortlaut sind im Volltext nicht nachgeprüft; die Aussage „Faktor $2^d$ für das Halbieren“ ist eine Rechnung, keine Aussage der Quelle);
Kolouri u. a. (2019), *Generalized sliced Wasserstein distances*, arXiv 1902.00434 (Gleichung 3); Fatras u. a. (2021), *Minibatch optimal transport distances; analysis and applications*, arXiv 2101.01792;
Boïté, Delon, Nadjahi (2026), *Expected batch optimal transport plans and consequences for flow matching*, arXiv 2605.12174 (Preprint; nur die Zusammenfassung gelesen); Feydy u. a. (2019), *Interpolating between optimal transport and MMD using Sinkhorn divergences*, arXiv 1810.08278 (Definition der Sinkhorn-Divergenz).
Die Formel $\mathbb{E}\lVert X - X'\rVert = 2\Gamma((d+1)/2)/\Gamma(d/2)$ und die Gauß-Formel für $W_2$ sind Lehrstoff und hier nachgerechnet.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html)."
)
