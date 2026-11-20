"""Fig. 1 del running example (Sección 2, Preliminaries): la figura principal. Instancia
el diagrama de fig1_retime.tex con los dos casos del par: el prefijo común a la
izquierda y un carril por caso (A arriba, W abajo) con lo que lee la política y su
decisión. Cada caja del estado lleva el borde del color de su nivel (riesgo, efecto,
nativo), los mismos colores que las Figs. 3, 5 y 6.

Escribe out/fig1_running_example.tex (el entorno figure, para \\input en el paper;
necesita tikz con arrows.meta y el color citehl que ya define main.tex) y una vista
previa compilada en PDF y PNG.
"""

import subprocess

from common import N_RESOURCES, OUT, activity, goal, load

d = load()
A, W = d["cards"]["A"], d["cards"]["W"]


def pct(x):
    return f"{100 * x:.0f}\\%"


def eur(v):
    return f"\\euro{{}}{int(float(v)):,}"


def lane(tag, c, y):
    ca, st = c["case_attributes"], c["state"]
    acts = c["action"] == "intervene"
    steps = int(round(c["prefix_nr"] / st["relative_position"]))
    lg = goal(ca["LoanGoal"])
    what = f"{lg} loan" if lg != "not specified" else "no loan goal given"
    dq = f"{c['dq']:+.2f}".replace("-", "$-$")
    dec = r"ACT NOW\\(second offer)" if acts else r"WAIT\\(no offer now)"
    p = c["p_act"]
    return rf"""
% --- caso {tag}: {c['case_id']} ---
\node[box, text width=2.9cm] (attr{tag}) at (3.85,{y}) {{\textbf{{{tag}}}: {what}, {eur(ca['RequestedAmount'])}\\{c['trace'][-1]['open_cases']:,} applications open}};
\node[box, draw=riskc, line width=0.9pt, text width=5.3cm] (risk{tag}) at (8.4,{y + 0.6}) {{risk: $r$ = {c['r']:.2f}, predicts failure}};
\node[box, draw=effectc, line width=0.9pt, text width=5.3cm] (eff{tag}) at (8.4,{y}) {{failure {c['pU']:.2f} without offer, {c['pT']:.2f} with}};
\node[box, text width=5.3cm] (nat{tag}) at (8.4,{y - 0.6}) {{step {c['prefix_nr']} of about {steps}, {int(round(st['available_resources']))} of {N_RESOURCES} staff free}};
\node[box, text width=2.35cm] (tim{tag}) at (12.85,{y}) {{timing model\\act {pct(p)}, wait {pct(1 - p)}\\$\Delta Q$ = {dq}}};
\node[box, fill={'posc' if acts else 'negc'}, text=white, font=\scriptsize\bfseries, text width=2.15cm] (dec{tag}) at (15.45,{y}) {{{dec}}};
\draw[->] (prefix.east) -- (attr{tag}.west);
\draw[->] (attr{tag}.east) -- (5.55,{y});
\draw[->] (11.25,{y}) -- (tim{tag}.west);
\draw[->] (tim{tag}.east) -- (dec{tag}.west);
"""


prefix = r"\\".join(f"{i}\\ {activity(e['activity'])}" for i, e in enumerate(A["trace"], 1))
yA, yW = 1.25, -1.25
pic = rf"""\begin{{tikzpicture}}[
  every node/.style={{align=center, font=\scriptsize}},
  box/.style={{draw, rounded corners, inner sep=2.5pt, minimum height=0.5cm}},
  >=Stealth
]
% el estado s_t de cada carril, resaltado como en fig1_retime.tex
\fill[citehl, rounded corners] (5.55,{yA - 0.98}) rectangle (11.25,{yA + 0.98});
\fill[citehl, rounded corners] (5.55,{yW - 0.98}) rectangle (11.25,{yW + 0.98});
\node[font=\scriptsize\itshape] at (8.4,{yA + 1.22}) {{state $s_t$ the timing model reads}};
\node[box, text width=3.7cm] (prefix) at (0.1,0) {{\textbf{{same prefix}}\\[1pt]{prefix}}};
{lane('A', A, yA)}
{lane('W', W, yW)}
% cabeceras
\node[font=\scriptsize\itshape] at (0.1,{yA + 1.22}) {{case so far (both)}};
\node[font=\scriptsize\itshape] at (3.85,{yA + 1.22}) {{case attributes}};
\node[font=\scriptsize\itshape] at (12.85,{yA + 1.22}) {{timing model}};
\node[font=\scriptsize\itshape] at (15.45,{yA + 1.22}) {{decision}};
% leyenda de niveles
\node[font=\scriptsize] at (8.4,{yW - 1.3}) {{box border:\enspace
  \textcolor{{riskc}}{{\rule{{6pt}}{{6pt}}}} from the risk model \enspace
  \textcolor{{effectc}}{{\rule{{6pt}}{{6pt}}}} from the effect model \enspace
  \rule{{6pt}}{{6pt}} native (from the case)}};
\end{{tikzpicture}}"""

colors = r"""\definecolor{riskc}{HTML}{7C3AED}
\definecolor{effectc}{HTML}{0F766E}
\definecolor{posc}{HTML}{C2410C}
\definecolor{negc}{HTML}{1D4ED8}"""

figure = rf"""\begin{{figure}}[!htb]
\centering
{colors}
\resizebox{{\columnwidth}}{{!}}{{%
{pic}%
}}
\caption{{Running example: two applications with the same three events and a similar risk. The timing model sends a second offer to A and waits on W}}
\Description{{Two lanes share one box on the left with the three events both applications went through: application created, submitted by the customer, staff follows up the lead. In the upper lane, application A, a car loan of 9,000 euros, has a risk of {A['r']:.2f}, a failure chance of {A['pU']:.2f} without a further offer and {A['pT']:.2f} with one, and {int(A['state']['available_resources'])} of 3 staff free; the timing model acts with {pct(A['p_act'])} and the decision is act now. In the lower lane, application W, with no loan goal given, of 18,500 euros, has a risk of {W['r']:.2f}, failure chances of {W['pU']:.2f} without and {W['pT']:.2f} with an offer, and {int(W['state']['available_resources'])} of 3 staff free; the timing model acts with only {pct(W['p_act'])} and the decision is wait.}}
\label{{fig:running}}
\end{{figure}}
"""
(OUT / "fig1_running_example.tex").write_text(figure)
print("->", OUT / "fig1_running_example.tex")

standalone = rf"""\documentclass[border=4pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta}}
\usepackage{{eurosym}}
\definecolor{{citehl}}{{RGB}}{{255,224,102}}
{colors}
\begin{{document}}
{pic}
\end{{document}}
"""
pv = OUT / "_preview"
pv.mkdir(exist_ok=True)
(pv / "fig1_preview.tex").write_text(standalone)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig1_preview.tex"], cwd=pv, capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-3000:])
    raise SystemExit("pdflatex falló")
(pv / "fig1_preview.pdf").replace(OUT / "fig1_running_example.pdf")
subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", str(OUT / "fig1_running_example.pdf"), str(OUT / "fig1_running_example")], check=True)
print("->", OUT / "fig1_running_example.pdf")
