"""Plotly-Figuren der Demo (alle Achsen fest und gesperrt, Beschriftungen als Annotationen)."""

import math

import numpy as np
import plotly.graph_objects as go

import dbd_constants as C
import dbd_wasserstein as W

_KARTE = dict(margin=dict(l=60, r=30, t=30, b=55), height=380, legend=dict(orientation="h", yanchor="top", y=-0.2, x=0), template="plotly_white")


def _achsen(fig, x_titel, y_titel, x_range, y_range, x_log=True, y_log=True):
    fig.update_xaxes(title_text=x_titel, type="log" if x_log else "linear", range=list(x_range), fixedrange=True)
    fig.update_yaxes(title_text=y_titel, type="log" if y_log else "linear", range=list(y_range), fixedrange=True)
    return fig


def _tickwerte(ns):
    return dict(tickmode="array", tickvals=list(ns), ticktext=[f"{n:,}".replace(",", " ") for n in ns])


def build_exp1(r: dict, referenz: dict | None) -> go.Figure:
    """W1 durch Maßstab gegen n (log-log): die gewählte Dimension live (Fehlerbalken: Standardfehler über die Wiederholungen), Anpassungsgerade, Bezugsreihen der anderen Dimensionen."""
    fig = go.Figure()
    ns = np.array(r["ns"], dtype=float)
    annot = []
    if referenz:
        for d_txt, ref in referenz.items():
            if int(d_txt) == r["d"]:
                continue
            fig.add_trace(go.Scatter(x=ref["ns"], y=ref["verhaeltnis"], mode="lines", line=dict(color=C.COLOR_REF, width=1.5), hoverinfo="skip", showlegend=False))
            annot.append(dict(x=math.log10(ref["ns"][-1]), y=math.log10(ref["verhaeltnis"][-1]), text=f"d = {d_txt}", showarrow=False, xanchor="left", xshift=4, font=dict(size=10, color="gray")))
    w1 = r["w1"] / r["massstab"]
    se = w1.std(axis=1, ddof=1) / math.sqrt(w1.shape[1])
    fig.add_trace(go.Scatter(x=ns, y=r["verhaeltnis"], mode="markers+lines", name=f"d = {r['d']}, {r['wdh']} Wiederholungen", line=dict(color=C.COLOR_MAIN, width=3),
                             error_y=dict(type="data", array=se, visible=True), marker=dict(size=8)))
    a, b = W.potenzanpassung(ns, r["verhaeltnis"])
    fig.add_trace(go.Scatter(x=ns, y=a * ns ** b, mode="lines", name=f"Anpassung, Steigung {b:.2f}".replace(".", ","), line=dict(color=C.COLOR_FIT, dash="dash", width=2)))
    anker = r["verhaeltnis"][0] * 32 ** 0.5
    fig.add_trace(go.Scatter(x=ns, y=anker * ns ** -0.5, mode="lines", name="Steigung −1/2 (Anker bei n = 32)", line=dict(color="gray", dash="dot", width=1.5)))
    for ziel in (0.10, 0.05):
        fig.add_hline(y=ziel, line=dict(color="gray", width=0.8, dash="dot"))
        annot.append(dict(x=math.log10(25), y=math.log10(ziel), text=f"{int(ziel * 100)} %", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=10, color="gray")))
    _achsen(fig, "n (Punkte je Stichprobe)", "W1 durch mittleren Paarabstand", (math.log10(24), math.log10(1900)), (math.log10(0.03), math.log10(1.0)))
    fig.update_xaxes(**_tickwerte(r["ns"]))
    fig.update_layout(annotations=annot, **_KARTE)
    return fig


def build_exp2_wahrheit(tabelle: list[dict]) -> go.Figure:
    """d = 1: Zwei-Stichproben-W1 und Abstand zur Wahrheit gegen n (log-log)."""
    ns = [z["n"] for z in tabelle]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[z["zwei"] for z in tabelle], mode="markers+lines", name="Zwei Stichproben", line=dict(color=C.COLOR_MAIN, width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=ns, y=[z["wahrheit"] for z in tabelle], mode="markers+lines", name="Stichprobe gegen Wahrheit N(0, 1)", line=dict(color=C.COLOR_TRUTH, width=3), marker=dict(size=8)))
    _achsen(fig, "n (Punkte je Stichprobe)", "W1 (d = 1)", (math.log10(24), math.log10(1400)), (math.log10(0.02), math.log10(0.6)))
    fig.update_xaxes(**_tickwerte(ns))
    fig.update_layout(**_KARTE)
    return fig


def build_zwei_punkt() -> go.Figure:
    """Zwei-Punkt-Verteilung: Verhältnis Zwei-Stichproben-Abstand zu Abstand zur Wahrheit gegen N (exakt), Grenzwert √2."""
    ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[W.zwei_punkt_verhaeltnis(N) for N in ns], mode="markers+lines", name="exakt (Binomialformeln)", line=dict(color=C.COLOR_MAIN, width=3), marker=dict(size=8)))
    fig.add_hline(y=math.sqrt(2), line=dict(color=C.COLOR_FIT, dash="dash", width=1.5))
    fig.update_layout(annotations=[dict(x=math.log10(2), y=math.sqrt(2), text="√2", showarrow=False, yanchor="bottom", xanchor="left", font=dict(color=C.COLOR_FIT))])
    _achsen(fig, "N (Punkte je Stichprobe, Zwei-Punkt-Verteilung)", "Zwei Stichproben / Wahrheit", (math.log10(1.6), math.log10(1400)), (1.38, 1.52), y_log=False)
    fig.update_xaxes(**_tickwerte([2, 8, 32, 128, 512, 1024]))
    fig.update_layout(**{**_KARTE, "annotations": fig.layout.annotations})
    return fig


def build_minibatch(r: dict, k_gewaehlt: int) -> go.Figure:
    """Verschoben: V gegen k mit wahrem W2² = 1; gleiche Verteilung: Basislinie gegen k mit Zufallspaarung 2d und dem Zwei-Stichproben-W2² (n = k)."""
    ks = [z["k"] for z in r["zeilen"]]
    fig = go.Figure()
    annot = []
    if r["form"] == "verschoben":
        v = np.array([z["v"] for z in r["zeilen"]])
        se = np.array([z["v_se"] for z in r["zeilen"]])
        fig.add_trace(go.Scatter(x=ks, y=v, mode="markers+lines", name="V = Minibatch-Wert / W2²", line=dict(color=C.COLOR_MAIN, width=3), error_y=dict(type="data", array=se, visible=True), marker=dict(size=8)))
        fig.add_hline(y=1.0, line=dict(color=C.COLOR_FIT, dash="dash", width=1.5))
        annot.append(dict(x=math.log10(7), y=1.0, text="wahres W2² (V = 1)", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=C.COLOR_FIT, size=11)))
        zg = next(z for z in r["zeilen"] if z["k"] == k_gewaehlt)
        annot.append(dict(x=math.log10(k_gewaehlt), y=zg["v"], text=f"k = {k_gewaehlt}", showarrow=True, ax=0, ay=-35, font=dict(size=11)))
        _achsen(fig, "k (Punkte je Batch)", "V", (math.log10(6), math.log10(180)), (0.8, 4.4), y_log=False)
    else:
        m = np.array([z["mittel"] for z in r["zeilen"]])
        se = np.array([z["se"] for z in r["zeilen"]])
        fig.add_trace(go.Scatter(x=ks, y=m, mode="markers+lines", name="Minibatch-Wert, P = Q (wahrer Abstand 0)", line=dict(color=C.COLOR_MAIN, width=3), error_y=dict(type="data", array=se, visible=True), marker=dict(size=8)))
        if "zwei_stichproben" in r:
            fig.add_trace(go.Scatter(x=ks, y=[z["mittel"] for z in r["zwei_stichproben"]], mode="markers", name="Zwei Stichproben, W2² mit n = k", marker=dict(symbol="diamond-open", size=11, color=C.COLOR_TRUTH, line=dict(width=2))))
        fig.add_hline(y=r["zufallspaarung"], line=dict(color=C.COLOR_FIT, dash="dash", width=1.5))
        annot.append(dict(x=math.log10(7), y=r["zufallspaarung"], text=f"zufällige Paarung ({r['zufallspaarung']:.0f})".replace(".", ","), showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=C.COLOR_FIT, size=11)))
        zg = next(z for z in r["zeilen"] if z["k"] == k_gewaehlt)
        annot.append(dict(x=math.log10(k_gewaehlt), y=zg["mittel"], text=f"k = {k_gewaehlt}", showarrow=True, ax=0, ay=-35, font=dict(size=11)))
        _achsen(fig, "k (Punkte je Batch)", "mittlere quadrierte Entfernung", (math.log10(6), math.log10(180)), (0, r["zufallspaarung"] * 1.12), y_log=False)
    fig.update_xaxes(**_tickwerte(ks))
    fig.update_layout(annotations=annot, **_KARTE)
    return fig


def build_exp3(r: dict) -> go.Figure:
    """Absolutes W1 gegen n (log-log) für die gewählten Daten in R^10, dazu die volle Dimension als Bezug und die Steigung −1/2."""
    ns = np.array(r["ns"], dtype=float)
    fig = go.Figure()
    voll = r.get("volle_dimension")
    if voll:
        fig.add_trace(go.Scatter(x=voll["ns"], y=voll["mittel"], mode="lines+markers", name="volle Dimension (Standardnormal in R^10, Bezug)", line=dict(color=C.COLOR_REF, width=2), marker=dict(size=6)))
    w1 = r["w1"]
    se = w1.std(axis=1, ddof=1) / math.sqrt(w1.shape[1])
    fig.add_trace(go.Scatter(x=ns, y=r["mittel"], mode="markers+lines", name=f"gewählte Daten, {r['wdh']} Wiederholungen", line=dict(color=C.COLOR_MAIN, width=3), error_y=dict(type="data", array=se, visible=True), marker=dict(size=8)))
    a, b = W.potenzanpassung(ns, r["mittel"])
    fig.add_trace(go.Scatter(x=ns, y=a * ns ** b, mode="lines", name=f"Anpassung, Steigung {b:.2f}".replace(".", ","), line=dict(color=C.COLOR_FIT, dash="dash", width=2)))
    anker = r["mittel"][0] * 32 ** 0.5
    fig.add_trace(go.Scatter(x=ns, y=anker * ns ** -0.5, mode="lines", name="Steigung −1/2 (Anker bei n = 32)", line=dict(color="gray", dash="dot", width=1.5)))
    _achsen(fig, "n (Punkte je Stichprobe)", "W1 (absolut)", (math.log10(24), math.log10(1400)), (math.log10(0.04), math.log10(5.0)))
    fig.update_xaxes(**_tickwerte(r["ns"]))
    fig.update_layout(**_KARTE)
    return fig
