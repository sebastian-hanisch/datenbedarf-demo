# Wie viele Daten braucht ein Abstand? Dimension, Struktur, Minibatch – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-datenbedarf-demo.streamlit.app/)** -->

Demo der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt an einer Annahme an, die die Demo „Abstand zwischen Verteilungen“ stehen lässt: *Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1.* Hier sind zwei **Stichproben** gegeben: der berechnete Abstand ist eine Zufallsgröße, und bei gleicher Verteilung ist er nicht null. Die Demo „Abstand zwischen Verteilungen“ zeigt Stichprobenrauschen nur an einem Beispiel (eine Dimension, ein wählbares n) und hebt die Annahme nicht auf; hier steht es im Mittelpunkt. Die Minibatch-Zuordnung in Experiment 2 ist dasselbe Zuordnungsproblem wie in „Vom Zuordnen zum Transportieren“, mit k Punkten.

## Warum dieses Problem

Wer zwei Datensätze mit einem Wasserstein-Abstand vergleicht, rechnet nicht den Abstand zweier Verteilungen, sondern den zweier **Stichproben**. Stammen beide aus derselben Verteilung, wäre die richtige Antwort null; gemessen wird der **Rauschboden** (englisch noise floor). Wie hoch er bei n Punkten liegt, hängt an der Dimension d: Die Demo misst im Standardnormal, wie schnell der Abstand zweier Stichproben aus derselben Verteilung mit n sinkt (Experiment 1), prüft, ob der Abstand der Stichproben der Abstand der Verteilungen ist, und was die Minibatch-Näherung kostet (Experiment 2), und zeigt, wie weit Struktur der Daten hilft (Experiment 3). Dazu kommt ein Abschnitt zum geschnittenen Wasserstein-Abstand (englisch sliced Wasserstein distance), der Verteilungen ordnet, aber etwas anderes misst. Die Demo stellt **eine** Annahme vor und prüft sie: „der Abstand der Stichproben ist der Abstand der Verteilungen“. Sie gilt bei d = 1 bis auf den Faktor √2 ≈ 1,41, bei d = 10 und n = 1 024 nicht.

## Modell

Zwei unabhängige Stichproben $X_1, \dots, X_n$ und $Y_1, \dots, Y_n$ aus dem Standardnormal $N(0, I_d)$. Formeln und Handrechnungen stehen im Expander „📐 Mathematische Formulierung“ der App.

- **Zwei-Stichproben-Abstand** (englisch two-sample distance): der Wasserstein-Abstand der zwei empirischen Verteilungen mit Gewicht $1/n$ je Punkt, also die **Zuordnung** (englisch assignment) mit den kleinsten mittleren Kosten. **W1** mit euklidischen Entfernungen, **W2** als Wurzel der kleinsten mittleren quadrierten Entfernung; in einer Dimension ist die sortierte Zuordnung die beste. Bei P = Q ist der wahre Abstand null: was gemessen wird, ist der Rauschboden.
- **Maßstab:** der mittlere Paarabstand zweier unabhängiger Punkte, $\mathbb{E}\lVert X - X'\rVert = 2\Gamma((d+1)/2)/\Gamma(d/2)$ (in der Demo mit `math.lgamma`); W1 wird durch ihn geteilt.
- **Abstand zur Wahrheit** wird nur bei d = 1 berechnet: $W_1(\hat P_n, N(0,1)) = \int \lvert F_n - \Phi\rvert$, exakt stückweise. Für d ≥ 2 gibt es keine einfache geschlossene Berechnung (die eindimensionale Struktur fehlt), die Demo rechnet ihn dort nicht; es steht der Zwei-Stichproben-Abstand stellvertretend, das ist eine Voraussetzung, nicht gemessen.
- **Steigung** (englisch slope): Kleinste-Quadrate-Gerade von log W1 gegen log n über die Mittelwerte; „alle n“ heißt n = 32, 64, …, 1 024, „letzte drei n“ heißt 256, 512 und 1 024. Nach Fournier und Guillin (Satz 1, gelesen) ist die obere Schranke für W1 bei genug Momenten $n^{-1/2}$ bei d = 1, $n^{-1/2}\log(1+n)$ bei d = 2 und $n^{-1/d}$ ab d = 3. Die Schranke gilt für den Abstand einer Stichprobe zur Verteilung; für den Abstand zweier Stichproben aus derselben Verteilung folgt dieselbe obere Schranke über die Dreiecksungleichung, $W_1(\hat P_n, \hat Q_n) \le W_1(\hat P_n, P) + W_1(P, \hat Q_n)$. Die Messung liegt bei n = 32 bis 1 024 und ist **vorasymptotisch**.
- **Teilstichproben-Zuordnung** (englisch minibatch assignment, hier „Minibatch“): das Zuordnungsproblem zwischen je k zufällig gezogenen Punkten aus P und aus Q, Wert = kleinste mittlere quadrierte Entfernung, gemittelt über die Zufalls-Seeds; V = Wert durch das wahre W2² (P gegen Q = N(2 e₁, 1,5² I): W2² = 4 + 0,25 d, also 4,5 bei d = 2 und 6,5 bei d = 10).
- **Zählweise:** Zähleinheit jeder Mittelung ist die **Wiederholung** (Zufalls-Seed): zwei neue Stichproben; im Minibatch ein Batch je Zufalls-Seed. Die Reihen dieser README benutzen den Zufalls-Seed 2026, unabhängig vom Regler (die App rechnet mit dem Zufalls-Seed des Reglers).
- **Zufall:** Hausgenerator SplitMix64 (`dbd_rng.py`, Kopie der Vorlage); die Ströme sind vektorisiert (`dbd_zufall.py`): die gleichverteilten Zahlen sind bitgleich zum Hausgenerator, die Normalzahlen sind Box-Muller mit numpy-Funktionen (in der letzten Binärstelle plattformabhängig, ohne Einfluss auf die genannten Stellen; der Hausgenerator allein rechnet rund 17 000 Normalzahlen je Sekunde, die vektorisierte Fassung rund 22 Millionen; Zeitmessung eines unbelasteten Rechners, bei Last entsprechend weniger, ein Hinweis, kein Test. Experiment 1 bei d = 10 mit 30 Wiederholungen und n bis 1 024 braucht 1 209 600 Normalzahlen, mit dem Hausgenerator allein rund 70 Sekunden).

## Methodik

- **Zuordnung und Maße** (`dbd_wasserstein.py`): `scipy.optimize.linear_sum_assignment` (die einzige scipy-Funktion der Laufzeit), mit der eigenen Ungarischen Methode (`dbd_zuordnung.py`, Kopie aus der Demo „Vom Zuordnen zum Transportieren“) als Rückfall bei fehlendem scipy; W1, W2², Maßstab, Abstand zur Wahrheit, Zwei-Punkt-Formeln in Brüchen.
- **Reihen** (`dbd_sweeps.py`): Stichproben der Familien (Standardnormal, Gerade mit Rauschen, zwei Blöcke), Reihen über n, nötiges n, Extrapolation, Minibatch.
- **Sliced** (`dbd_sliced.py`): Gauß-Familien, geschlossene Wahrheit (Bures), Quadratur über die Kugel, Rangkorrelationen.
- **Auswertung** (`dbd_evaluation.py`): Einstellungen, Formatierung, Verdikte als reine Funktionen.
- **Vorgerechnet** (`tools/messreihen.py` → `tools/messreihen.json`, `dbd_vorgerechnet.py`): Bezugsreihen je Dimension, Extrapolation, Streuung der Steigung, Struktur mit 100 Wiederholungen, Minibatch mit 2 000 Zufalls-Seeds, Tabelle zu sliced. Die Experimente 1 bis 3 und der Minibatch-Teil rechnen live (mit Zwischenspeicher).
- **Rückfall ohne scipy:** die eigene Ungarische Methode rechnet denselben Wert (Test), ist aber langsamer. Zeitmessungen aus zwei Läufen (ein Kern, Python 3.14, Last auf dem Rechner verschieden; ein Hinweis, kein Test), je eine W1-Rechnung mit Ziehen der Stichproben und Kostenmatrix: n = 1 024 in d = 10 etwa 1,0 bis 1,3 s gegen 0,09 bis 0,10 s mit scipy, bei zwei Atomen (viele Gleichstände) 4,3 bis 6,0 s gegen 0,04 s (etwa 107- bis 167-mal langsamer); die ganze Reihe von Experiment 1 bei d = 10 (30 Wiederholungen, n bis 1 024) 37 bis 57 s gegen 3,3 bis 4,3 s. Nur für die Zuordnung (ein Lauf, bestes von mehreren Versuchen) ist der Rückfall bei Standardnormal-Stichproben mit d = 2 und 10 und n = 128 bis 2 048 etwa 11- bis 44-mal langsamer als scipy (bei n = 128 bis 256 mit 26- bis 44-mal am stärksten, bei n = 1 024 und mehr 11- bis 18-mal), bei zwei Atomen mit n = 128 bis 1 024 mehr als 700-mal. Die App weist auf den Rückfall hin; bei fehlendem scipy ist n bis 256 empfohlen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus dem Code der Demo (Zufalls-Seed 2026, `tools/messreihen.py`); die Tabellen und Sätze unten werden aus `tools/messreihen.json` gebildet und stehen wörtlich in `tests/test_claims.py` (Zahl → Test: Abschnitt „Zahl → Test“ am Ende dieses Abschnitts).

### Experiment 1: Steigung gegen Dimension, nötiges n

W1 durch den Maßstab, Standardnormal, n = 32 bis 1 024, Mittel über die Wiederholungen. **n für 10 % und 5 %**: Potenzanpassung a · n^b des Verhältnisses (W1 durch Maßstab) bis zum Wert 10 % beziehungsweise 5 %; ein Stern heißt Extrapolation über n = 1 024 hinaus, nicht gemessen. **Halbierungsfaktor:** 2^(1/|Steigung|).

| d | Wiederholungen | Steigung (alle n) | Steigung (letzte drei n) | W1 durch Maßstab, n = 32 | n = 1 024 | n für 10 % | n für 5 % | Halbierungsfaktor |
|---|---|---|---|---|---|---|---|---|
| 1 | 200 | −0,493 | −0,510 | 28,6 % | 4,9 % | 256 | 1 040* | 4 |
| 2 | 30 | −0,422 | −0,416 | 36,6 % | 8,4 % | 686 | 3 540* | 5 |
| 3 | 30 | −0,309 | −0,338 | 43,3 % | 14,6 % | 3 720* | 35 100* | 9 |
| 4 | 30 | −0,242 | −0,254 | 49,4 % | 21,2 % | 23 300* | 4,1e5* | 17 |
| 5 | 30 | −0,201 | −0,210 | 54,1 % | 27,0 % | 1,5e5* | 4,6e6* | 32 |
| 10 | 30 | −0,113 | −0,114 | 67,5 % | 45,7 % | 7,1e8* | 3,3e11* | 463 |

Bei n = 1 024 liegt der Zwei-Stichproben-Abstand bei d = 10 bei 46 % des mittleren Paarabstands, bei d = 5 bei 27 % und bei d = 1 bei 5 %; der wahre Abstand ist in allen Fällen null. Aus der gemessenen Steigung folgt (Rechnung, vorasymptotisch) ein Halbierungsfaktor 2^(1/|Steigung|) von 32 bei d = 5 und 463 bei d = 10; die Schranke mit Steigung −1/d gäbe 32 und 1024. Jede Zelle ist ein Mittel über 200 (d = 1) beziehungsweise 30 Wiederholungen und streut: Standardfehler der Zellen in Prozentpunkten bei n = 32 0,8 / 1,0 / 0,8 / 0,7 / 0,6 / 0,4 und bei n = 1 024 0,12 / 0,12 / 0,06 / 0,05 / 0,06 / 0,06 (d = 1 / 2 / 3 / 4 / 5 / 10). Beispiel d = 1, n = 32: die Zelle (28,6 %, Standardfehler 0,8) liegt rund 2,4 Standardfehler über der Erwartung 26,8 % (Mittel über 100 000 weitere Wiederholungen mit einem anderen Zufallsgenerator), also am oberen Rand dessen, was beim Zufalls-Seed 2026 vorkommt; die Steigung bei d = 1 ist davon kaum betroffen. Der Stern bei d = 1 (n für 5 % = 1 040*) gilt für die Anpassung: sie erreicht 5 % erst bei n = 1 040, 5,05 % bei n = 1 024, während der gemessene Mittelwert bei n = 1 024 mit 4,9 % schon unter 5 % liegt (Standardfehler 0,12 Prozentpunkte); der Abstand zum Raster ist kleiner als diese Streuung. Bei d = 1 ist die Steigung etwa −0,5, nicht −1. Die Formel n^(−1/d) ist die obere Schranke ab d = 3; bei d = 1 und 2 gelten die oben genannten Fälle.

**Extrapolation um den Faktor 4 geprüft:** Verhältnis der Messung bei n = 2 048 und 4 096 zur Vorhersage der Anpassung aus n = 32 bis 1 024 (Klammer: Standardfehler des Messwerts).

| d | Wiederholungen | n = 2 048 | n = 4 096 |
|---|---|---|---|
| 1 | 200 | 0,946 (± 2,3 %) | 1,031 (± 2,9 %) |
| 2 | 10 | 0,949 (± 1,8 %) | 0,974 (± 2,4 %) |
| 3 | 10 | 0,967 (± 0,7 %) | 0,953 (± 0,3 %) |
| 4 | 10 | 0,992 (± 0,4 %) | 0,985 (± 0,3 %) |
| 5 | 10 | 0,997 (± 0,2 %) | 0,994 (± 0,2 %) |
| 10 | 10 | 1,005 (± 0,1 %) | 1,009 (± 0,1 %) |

Die größte Abweichung der Messung von der Vorhersage beträgt 5,4 % (d = 1, n = 2 048); in allen zwölf Zellen liegt die Messung zwischen 0,946 und 1,031 der Vorhersage, in 9 von 12 darunter. Die Werte mit Stern jenseits von n = 4 096 reichen bei d = 10 bis zu 8,5 Größenordnungen über n = 1 024 hinaus (für 5 %); sie sind Rechenbeispiele, keine Messung.

**Streuung der Steigung von Satz zu Satz** (unabhängige Sätze von je 30 Wiederholungen, Zufalls-Seeds 3000 bis 3019 bei d = 1 und 3000 bis 3009 sonst):

| Daten | unabhängige Sätze von je 30 Wiederholungen | Steigung alle n, Mittel über die Sätze | Steigung letzte drei n, Mittel über die Sätze |
|---|---|---|---|
| d = 1, Standardnormal | 20 | −0,489 (Standardabweichung 0,028) | −0,503 (Standardabweichung 0,082; −0,71 bis −0,38) |
| Gerade in R^10 | 10 | −0,475 (Standardabweichung 0,019) | −0,500 (Standardabweichung 0,060; −0,64 bis −0,41) |
| d = 10, Standardnormal | 10 | −0,114 (Standardabweichung 0,002) | −0,110 (Standardabweichung 0,002; −0,11 bis −0,11) |

Die Steigung über die letzten drei n streut bei 30 Wiederholungen von Satz zu Satz mit einer Standardabweichung von 0,08 bei d = 1 und 0,06 bei der Geraden in R^10 (über alle n 0,03 und 0,02), bei d = 10 nur 0,002.

### Experiment 2: Rauschboden (Zwei Stichproben) und Minibatch

**Zwei Stichproben, d = 1:** Zwei-Stichproben-W1 gegen den Abstand einer Stichprobe zur Wahrheit N(0, 1) (Quadratur).

| n | Zwei Stichproben (W1) | Stichprobe gegen Wahrheit (W1) | Verhältnis |
|---|---|---|---|
| 32 | 0,3226 | 0,2335 | 1,38 |
| 64 | 0,2136 | 0,1579 | 1,35 |
| 128 | 0,1594 | 0,1119 | 1,42 |
| 256 | 0,1130 | 0,0818 | 1,38 |
| 512 | 0,0829 | 0,0587 | 1,41 |
| 1 024 | 0,0558 | 0,0388 | 1,44 |

Mittel der sechs Verhältnisse 1,40, je n zwischen 1,35 und 1,44 (200 Wiederholungen je n, d = 1, Standardnormal); der Grenzwert ist √2 = 1,414. Der Faktor ist für d ≥ 2 nicht gemessen. Im Zwei-Punkt-Fall ist das Verhältnis exakt 1,5 bei N = 2, 1,4583 bei N = 4 (35/24), 1,4197 bei N = 32 und 1,4144 bei N = 1 024, fallend gegen √2. Im Grenzwert ist der Faktor √2, weil beide Abstände gaußsche Prozesse sind und der Zwei-Stichproben-Prozess die doppelte Varianz hat (Rechnung, nicht gemessen).

**Minibatch**, P = N(0, I_d), Q = N(2 e₁, 1,5² I_d), 2 000 Zufalls-Seeds je Zelle, Zufalls-Seed 2026; V = Minibatch-Wert durch wahres W2²; die letzte Spalte ist die Basislinie zwischen gleichen Verteilungen (reiner Rauschboden, derselbe wie bei zwei Stichproben mit n = k).

| k | V bei d = 10 (verschoben und gestreckt) | Batches mit V unter 1 | Wert bei P = Q, d = 10 (wahrer Abstand 0; Zufallspaarung 20) |
|---|---|---|---|
| 8 | 3,962 ± 0,014 | 0,0 % | 12,857 ± 0,043 |
| 16 | 3,481 ± 0,009 | 0,0 % | 10,801 ± 0,025 |
| 32 | 3,125 ± 0,006 | 0,0 % | 9,190 ± 0,015 |
| 64 | 2,807 ± 0,004 | 0,0 % | 7,812 ± 0,009 |
| 128 | 2,533 ± 0,003 | 0,0 % | 6,662 ± 0,006 |

Bei d = 10 und k = 128 ist V = 2,533 ± 0,003 und der Wert zwischen gleichen Verteilungen 6,66 ± 0,01 statt 0 (2 000 Zufalls-Seeds, Zufalls-Seed 2026); keiner der 2 000 Batches liegt bei d = 10 unter dem wahren W2² (Anteil 0,0 %). Die Basislinie ist per Definition der Zwei-Stichproben-Abstand zum Quadrat mit n = k.

| k | V bei d = 2 (verschoben und gestreckt) | Batches mit V unter 1 | Wert bei P = Q, d = 2 (wahrer Abstand 0; Zufallspaarung 4) |
|---|---|---|---|
| 8 | 1,458 ± 0,014 | 24,1 % | 1,381 ± 0,014 |
| 16 | 1,288 ± 0,010 | 27,1 % | 0,873 ± 0,007 |
| 32 | 1,198 ± 0,007 | 26,8 % | 0,544 ± 0,004 |
| 64 | 1,107 ± 0,005 | 32,5 % | 0,328 ± 0,002 |
| 128 | 1,057 ± 0,003 | 36,9 % | 0,195 ± 0,001 |

Bei d = 2 sind V − 1 und die Steigung von log(V − 1) zwischen k = 32 und 128 von Satz zu Satz unsicher (fünf Sätze von je 2 000 Zufalls-Seeds):

| Zufalls-Seed (Satz von 2 000) | V bei k = 32 | V bei k = 128 | Steigung von log(V − 1), k = 32 bis 128 | Batches V < 1, k = 32 | k = 128 |
|---|---|---|---|---|---|
| 2026 | 1,198 | 1,057 | −0,90 | 26,8 % | 36,9 % |
| 2027 | 1,179 | 1,063 | −0,75 | 29,5 % | 34,9 % |
| 2028 | 1,183 | 1,062 | −0,78 | 27,7 % | 34,8 % |
| 2029 | 1,182 | 1,064 | −0,75 | 28,6 % | 34,3 % |
| 2030 | 1,180 | 1,061 | −0,78 | 28,3 % | 35,3 % |

Über die fünf Sätze von je 2 000 Zufalls-Seeds liegt V bei d = 2 und k = 128 zwischen 1,057 und 1,064, bei k = 32 zwischen 1,179 und 1,198; die Steigung von log(V − 1) zwischen −0,90 und −0,75; der Anteil der Batches unter dem wahren W2² zwischen 26,8 % und 36,9 %. Ein Teil der einzelnen Batches liegt bei d = 2 unter dem wahren W2², im Mittel liegt V stets darüber (die Erwartung des Minibatch-Werts ist nach gemeinsamer Konvexität nie kleiner als W2²; im Zwei-Punkt-Beispiel exakt: V = 9/8 bei k = 1 und 35/32 bei k = 2, Handrechnung im Test).

### Experiment 3: Struktur

R^10, Zwei-Stichproben-W1 (absolut), n = 32 bis 1 024; zwei Blöcke bei −3 e₁ und +3 e₁ mit Streuung σ je Achse (σ = 0: zwei Atome), Gerade + Rauschen: Achse e₁ standardnormal, die neun anderen Achsen mit Streuung σ.

| Daten in R^10 | Steigung alle n / letzte drei n, 30 Wiederholungen | 100 Wiederholungen | W1 bei n = 32 / 1 024 (100 Wiederholungen) |
|---|---|---|---|
| Gerade (σ = 0) | −0,45 / −0,37 | −0,48 / −0,49 | 0,309 / 0,058 |
| Gerade + Rauschen σ = 0,1 | −0,17 / −0,14 | −0,19 / −0,15 | 0,506 / 0,265 |
| Gerade + Rauschen σ = 0,3 | −0,12 / −0,11 | −0,13 / −0,11 | 1,056 / 0,683 |
| Gerade + Rauschen σ = 1 (volle Dimension) | −0,11 / −0,11 | −0,11 / −0,11 | 2,947 / 1,996 |
| Zwei Atome (σ = 0) | −0,50 / −0,40 | −0,56 / −0,51 | 0,696 / 0,098 |
| Zwei Blöcke σ = 0,1 | −0,30 / −0,21 | −0,33 / −0,25 | 0,956 / 0,302 |
| Zwei Blöcke σ = 0,3 | −0,19 / −0,15 | −0,21 / −0,16 | 1,485 / 0,711 |
| Zwei Blöcke σ = 1 | −0,13 / −0,11 | −0,13 / −0,12 | 3,433 / 2,162 |

Die Gerade in R^10 hat mit 30 Wiederholungen die Steigung −0,446 (letzte drei n −0,372), mit 100 Wiederholungen −0,479 (−0,489); über 10 unabhängige Sätze von je 30 Wiederholungen liegt die Steigung bei −0,475 (alle n) und −0,500 (letzte drei n), die Kontrolle d = 1 bei −0,489 und −0,503: eine Gerade verhält sich wie d = 1. Zwei Atome im Abstand 6: exakt E W1 = 6 · C(2n, n)/4^n = 0,5961 bei n = 32 und 0,1058 bei n = 1 024, Steigung −0,499 (Binomialrechnung, kein Zufall); die gemessene Steigung mit 100 Wiederholungen ist −0,556, mit 30 Wiederholungen −0,498: das Rauschen der Wiederholungen, nicht ein anderer Wert. Schon kleine Streuung (σ = 0,1 bis 0,3) senkt die Steigung (alle n, 100 Wiederholungen) bei der Geraden auf −0,19 bis −0,13, bei zwei Blöcken auf −0,33 bis −0,21; bei σ = 1 (volle Dimension) liegt sie bei −0,11 (Gerade) und −0,13 (zwei Blöcke), bei d = 5 und d = 10 in der Reihe von Experiment 1 bei −0,20 und −0,11. Die Gerade erreicht diese Werte ab σ = 0,1 bis 0,3, zwei Blöcke mit σ = 0,1 (−0,33) noch nicht. Struktur hilft in dem Maß, in dem sie exakt ist (n = 32 bis 1 024, vorasymptotisch; nur vier Werte von σ gemessen). Die Untergrenze n^(−1/2) bei getrenntem Träger (Fournier und Guillin, Abschn. 1.2) ist eine Untergrenze, keine bessere Rate.

### Geschnittener Wasserstein-Abstand (sliced)

Referenz N(0, I_10), 40 Gauß-Verteilungen N(μ, Σ) in vier Familien zu je zehn Stärken m = 0,1; 0,2; …; 1,0 (S: Verschiebung um m; I: isotrope Streuung, Standardabweichung 1 + m; E: Streuung nur in einer Achse, Standardabweichung 1 + 3m; K: Gleichkorrelation mit ρ = m/2). Verhältnis sliced zu W2 (Population, geschlossen, Quadratur über die Kugel):

| Familie | Mittel über zehn Stärken | kleinster Wert | größter Wert |
|---|---|---|---|
| S Verschiebung | 0,316 | 0,316 | 0,316 |
| I isotrope Streuung | 0,316 | 0,316 | 0,316 |
| E Streuung in einer Achse | 0,210 | 0,172 | 0,236 |
| K Gleichkorrelation | 0,155 | 0,137 | 0,168 |

Das Verhältnis hängt an der Stärke und an ihrer Definition: bei anderen Skalen der Stärke ergeben sich andere Mittel. Bei Verschiebung und isotroper Streuung ist es bei jeder Stärke 1/√d (0,316 bei d = 10). Rangübereinstimmung mit der geschlossenen Wahrheit (Stichproben mit n = 300, Zufalls-Seed 2026, 20 Richtungssätze je L):

| Abstand | Spearman | Kendall τ | Spearman je Familie |
|---|---|---|---|
| voller Abstand W2 aus Stichproben (eine Stichprobe je Fall, ohne Streuungsangabe) | 0,916 | 0,780 | 0,985 |
| sliced, Population (geschlossen) | 0,934 | 0,805 | 1,000 |
| sliced aus Stichproben, L = 1 | 0,631 | 0,486 | 0,707 |
| sliced aus Stichproben, L = 5 | 0,829 | 0,679 | 0,928 |
| sliced aus Stichproben, L = 20 | 0,910 | 0,766 | 0,974 |
| sliced aus Stichproben, L = 100 | 0,926 | 0,787 | 0,980 |
| sliced aus Stichproben, L = 500 | 0,928 | 0,789 | 0,980 |

Rauschboden des vollen W2 zwischen zwei Stichproben aus N(0, I_10) mit n = 300: 2,35 (Mittel über 5 Paare); die Wahrheit der 40 Fälle liegt zwischen 0,10 und 3,16. Sliced ordnet für die Population innerhalb jeder Familie perfekt (Spearman 1,000), aus Stichproben mit n = 300 und L = 500 Richtungen mit Spearman 0,980 je Familie, und über die vier Familien nur mit Spearman 0,93, weil es Streuung in einer Achse und Korrelation schwächer gewichtet als eine Verschiebung: es misst eine andere, schwächere Distanz. Die Rate von sliced selbst ist nicht gemessen.

### Befunde und Korrekturen: was die Messung gegenüber den üblichen Erwartungen zeigt

- **d = 1: Steigung etwa −0,5, nicht −1.** Die Formel „Rate n^(−1/d)“ steht nur mit „ab d = 3“; bei d = 1 und 2 gelten die Fälle nach Fournier und Guillin.
- **Faktor Zwei Stichproben gegen Wahrheit:** gemessen 1,40 im Mittel (je n zwischen 1,35 und 1,44), Grenzwert √2; nur bei d = 1 und nur für das Standardnormal.
- **Gerade:** die Steigung ist die von d = 1, aber mit 30 Wiederholungen streut sie über die letzten drei n von Satz zu Satz mit einer Standardabweichung von 0,06; ein einzelner Wert wie −0,37 liegt rund 2,1 Standardabweichungen neben dem Mittel −0,50 der 10 Sätze, ist also mit dem Rauschen vereinbar und kein Gegenbeispiel (100 Wiederholungen: −0,49).
- **Zwei Atome:** die Steigung ist exakt −0,499; eine Messreihe mit 100 Wiederholungen ergibt −0,56 (Rauschen der Wiederholungen).
- **Minibatch bei d = 2:** V bei k = 128 liegt in fünf Sätzen von je 2 000 Zufalls-Seeds zwischen 1,057 und 1,064; bei k = 32 weichen die Sätze um bis zu 0,019 voneinander ab, die Steigung von log(V − 1) um bis zu 0,14. Die Zahl wird mit ihrer Zahl der Zufalls-Seeds genannt, nie allein.

### Zahl → Test

| Zahl oder Aussage | Test (Band oder exakt) |
|---|---|
| alle Tabellen und Sätze dieses Abschnitts | `tests/test_claims.py::test_tabellen_und_saetze_der_readme` (wörtlich aus `tools/messreihen.json`) |
| Zellen der Tabellen neu gerechnet | `test_claims.py::test_vorgerechnete_zellen_stimmen_beim_neu_rechnen` (Band 1e-6 relativ), `test_quellen_pruefsumme_der_messreihen` (die Datei gehört zum Code) |
| Aussagen „in k von m Zellen“ | `test_claims.py::test_zaehlaussagen_der_readme` |
| Steigung −0,499 der zwei Atome, Zwei-Punkt-Formeln | `tests/test_wasserstein.py`, `tests/test_evaluation.py::test_exakte_steigung_der_zwei_atome_ist_minus_0_499` (exakt in Brüchen) |
| Abstand zur Wahrheit, Maßstab, sliced-Verhältnisse | `tests/test_oracle_datenbedarf.py` (Quadratur, Trapezsumme, Formel mit `gammaln`, Matrixwurzel) |
| Verdikt je Schnellstart-Szenario | `tests/test_app.py::test_preset_*` |

## Ehrliche Grenzen

- **Vorasymptotisch:** n = 32 bis 1 024; die Steigungen sind keine Raten für n → ∞ (bei d = 10 gemessen −0,11 gegen −0,1 der Schranke). Der Halbierungsfaktor ist eine Rechnung aus der gemessenen Steigung.
- **Nur Standardnormal** (und die Strukturen in R^10, die Gauß-Familien von sliced): Gleichverteilung im Würfel, Mischungen, schwere Ränder und P ≠ Q beim Zwei-Stichproben-Abstand sind nicht gemessen (Fournier und Guillin nennen bei schweren Rändern eine andere Untergrenze).
- **d = 3 und 4** sind in den Reihen enthalten und gemessen; sie stehen in den Tabellen.
- **Abstand zur Wahrheit** nur bei d = 1 berechnet (für d ≥ 2 nicht berechnet).
- **Streuung der Steigung:** siehe Tabelle in Experiment 1; ein einzelner Satz von Wiederholungen ist kein genauer Wert.
- **Minibatch:** eine isotrope Gauß-Familie, exakte Zuordnung; anisotrop, Mischungen, mit Entropie und stetig gegen stetig sind nicht gemessen. Zur Aussage „Verzerrung nicht wachsend in k, O(1/k) bei kompaktem Träger“ (Boïté, Delon, Nadjahi 2026, Preprint; nur die Zusammenfassung gelesen, die genaue Form steht im Volltext, der nicht geprüft ist) macht die Demo keine Aussage; die Messung ist Gauß, nicht kompakt.
- **Sinkhorn-Divergenz** und andere entropische Schätzer mit ε-abhängigen Konstanten sind nicht gemessen. Zu Zwei-Stichproben-Tests mit W1 in hoher Dimension macht die Demo keine Aussage (die Voraussetzungen der Quelle sind hier nicht geprüft).
- **Sliced:** nur Gauß, d = 10, n = 300; die Stichprobenrate von sliced und sein Rauschboden sind nicht gemessen.
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).

## Verifikation

- **Testzahl:** 200 Tests (`pytest --collect-only -q tests`).
- **Handrechnungen:** drei Werte in einer Dimension (W1 = 1, W2² = 5/3, gleich der Aufzählung aller Zuordnungen und dem Integral der Verteilungsfunktionen), Zwei-Punkt-Verteilung in Brüchen (3/8, 35/128, 1/4, 3/16, Verhältnis 3/2 und 35/24, Aufzählung bis N = 12), zwei Atome in R^10, Punkte auf einer Geraden, Minibatch im Zwei-Punkt-Beispiel (1/2, 3/8, 9/2, 35/8).
- **Orakel** (`tests/test_oracle_datenbedarf.py`): Zuordnung gegen ein Linearprogramm (HiGHS), den Netzwerksimplex von `networkx` und die Aufzählung aller Zuordnungen; eindimensional gegen `scipy.stats.wasserstein_distance`; Maßstab gegen die `gammaln`-Formel und Monte-Carlo; Abstand zur Wahrheit gegen Quadratur und Trapezsumme; Quadratur über die Kugel gegen `scipy.integrate.quad`; Bures-Wert gegen die Matrixwurzel; Rangkorrelationen gegen `scipy.stats`; Minibatch-Basislinie gegen den Zwei-Stichproben-Abstand.
- **Eigenschaften** (`tests/test_eigenschaften.py`): Skalieren, Verschieben, Drehen, Vertauschen, Umordnen, W1 ≤ W2, Dreiecksungleichung, Gerade gleich eindimensional, zwei Atome gleich Δ·|Z − Z′|, mehr Daten senken den Abstand, Minibatch im Mittel über dem wahren W2².
- **Kopien:** `dbd_rng.py` und `dbd_zuordnung.py` sind Kopien (Test auf Gleichheit mit der Quelle); der Zufall ist bitgleich zum Hausgenerator (`tests/test_zufall.py`), auch in einem zweiten Prozess mit anderem Hash-Seed.
- **Oberfläche:** AppTest-Rauchtests (Standard, jedes Schnellstart-Szenario, Permalink, jede Datenart, jede Dimension, jeder Regler an Minimum und Maximum, Worst-Case mit allen Maxima, Rückfall ohne scipy), Footer-Test, Laufzeit-Importe (scipy steht in `requirements.txt`).
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Szenario hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf (`tests/test_layout.py`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `dbd_constants.py` | Regler-Auswahlen, Standardwerte, Presets, Farben |
| `dbd_evaluation.py` | Einstellungen, Formatierung, Auswertung und Verdikte |
| `dbd_layout.py` | Fester Kopf (Stylesheet) |
| `dbd_presets.py` | Permalink (Begrenzen, Einrasten), Schnellstart in der Seitenleiste, Zufalls-Seed-Button |
| `dbd_rng.py` | Hausgenerator SplitMix64 (Kopie der Vorlage) |
| `dbd_sliced.py` | Geschnittener Wasserstein-Abstand, Gauß-Familien, Rangkorrelationen |
| `dbd_sweeps.py` | Reihen: Stichproben, Struktur, Minibatch, Extrapolation |
| `dbd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `dbd_vorgerechnet.py` | liest `tools/messreihen.json`, Prüfsumme der Quellen |
| `dbd_wasserstein.py` | Zuordnung, W1, W2², Maßstab, Abstand zur Wahrheit, Zwei-Punkt-Formeln |
| `dbd_zufall.py` | vektorisierte Zufallsströme |
| `dbd_zuordnung.py` | Ungarische Methode (Kopie, Rückfall ohne scipy) |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; scipy nur für die Zuordnung) |
| `tools/messreihen.py`, `tools/messreihen.json` | die vorgerechneten Reihen und ihr Erzeuger |
| `tools/readme_bauen.py`, `tools/readme_vorlage.md` | setzt die Tabellen aus der Messdatei in diese README ein |
| `tests/` | Handrechnungen, Orakel, Eigenschaften, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Kopien |

## Bewusst nicht umgesetzt

- Gromov-Wasserstein, Wasserstein-DRO und Entscheidungsvergleiche; entropische Schätzer; andere Verteilungen als Standardnormal; Realdaten.
- Eine Testrate für Zwei-Stichproben-Tests mit W1; die Rate von sliced; die Minibatch-Rate bei kompaktem Träger.
- PDF-Export und Datei-Upload.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (nur `linear_sum_assignment`).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
