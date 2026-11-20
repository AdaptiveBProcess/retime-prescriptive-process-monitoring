"""Fig. 5 del running example: las tres explicaciones de una decisión (card de tres
niveles), para A o para W. Misma maqueta que figures/make_three_level.py del repo y
que la versión de entregables/figuras-ejemplo/scripts (etiquetas llanas, columna de
valores, signo menos tipográfico), sin las marcas rojas de revisión y leyendo las
cards del par (datos/cards_running_example.json).

Uso: python make_fig5.py [A|W]     (sin argumento: los dos)
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from common import INK, LEVEL, LINE, MUTED, NATIVE, NEG, POS, activity, goal, label, load, save, signed, state_label  # noqa: E402

# revisión: etiqueta más corta de Activity para que quepa en el panel (el texto del paper la nombra).
_label = label
def label(*a, **k):
    s = _label(*a, **k)
    return "alphabetically last activity" if s == "alphabetically last activity (Activity)" else s


# revisión: en el panel 3, lo nativo ("case" aquí) va en café, como en la Figura 1 del paper.
LEVEL = {**LEVEL, "case": NATIVE}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": LINE, "xtick.color": INK, "ytick.color": INK})
BORDER = "#000000"
SHOW_BAND = False  # la banda repetía la Tabla 1 y la Figura 1 del paper
FS = 1.12  # revisión: letra más grande, la figura va a media columna junto a la de W
TIMING_LIM = None  # revisión: misma escala del panel 3 en A y W (la de la antigua Figura 3), se fija en __main__
PAIR_TITLE = True  # revisión: A y W en una figura, con títulos (a)/(b) como la Figura 5 del paper


def draw(c, tag):
    acts = c["action"] == "intervene"
    st = c["state"]
    fig = plt.figure(figsize=(13.0, 6.3))

    def box(x, y, w, h, fc="white", ec=BORDER, lw=0.8, ls="-", z=1):
        fig.patches.append(patches.Rectangle((x, y), w, h, transform=fig.transFigure, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def arrow(x0, x1, y, ls="-", color=MUTED):
        fig.patches.append(patches.FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11,
                                                   color=color, lw=1.0, ls=ls, zorder=2))

    def two_lines(s):
        words = s.split()
        if len(words) < 2:
            return s, len(s)
        cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        a, b = " ".join(words[:cut]), " ".join(words[cut:])
        return f"{a}\n{b}", max(len(a), len(b))

    # --- banda de arriba: qué decisión es (quitada en la revisión: repite la Tabla 1 y la Figura 1) ---
    if SHOW_BAND:
        TOP0, TOP1 = 0.775, 0.985
        box(0.012, TOP0, 0.976, TOP1 - TOP0, fc="#f8fafc")
        fig.text(0.024, 0.955, f"Running example, case ${{\\it {tag}}}$: which decision point is explained?", fontsize=13.8 * FS, fontweight="bold",
                 color=INK, va="center")
        ca = c["case_attributes"]
        fig.text(0.024, 0.918, f"Case {c['case_id']} (BPIC2017 test split), right after its event {c['prefix_nr']}:  "
                 f"{ca['ApplicationType'].lower()}, goal {goal(ca['LoanGoal'])}, €{int(float(ca['RequestedAmount'])):,} requested",
                 fontsize=11.2 * FS, color=INK, va="center")
        y_tl, x, BOX_H = 0.835, 0.024, 0.086
        trace = [e["activity"] for e in c["trace"]]
        shown = trace if len(trace) <= 3 else [trace[0], None, trace[-1]]
        for a in shown:
            txt, n = two_lines(activity(a)) if a else (f"... {len(trace) - 2} more events ...", 18)
            w = 0.014 + 0.00808 * max(n, 8)
            box(x, y_tl - BOX_H / 2, w, BOX_H, fc="white" if a else "#f8fafc")
            fig.text(x + w / 2, y_tl, txt, fontsize=12.6 * FS, color=INK, ha="center", va="center", linespacing=1.25, style="normal" if a else "italic")
            arrow(x + w + 0.003, x + w + 0.02, y_tl)
            x += w + 0.023
        w_dp = 0.245
        box(x, y_tl - BOX_H / 2, w_dp, BOX_H, fc="#fef3c7", lw=1.4)
        fig.text(x + w_dp / 2, y_tl, "Decision point: intervene or wait?", fontsize=11.4 * FS, fontweight="bold", color=INK, ha="center", va="center")
        x_dec = x + w_dp
        arrow(x_dec + 0.004, x_dec + 0.03, y_tl, color="#f59e0b")
        w_d = 0.128
        box(x_dec + 0.033, y_tl - BOX_H / 2, w_d, BOX_H, fc=POS if acts else NEG)
        fig.text(x_dec + 0.033 + w_d / 2, y_tl, "timing model:\nINTERVENE" if acts else "timing model:\nWAIT", fontsize=10.6 * FS, fontweight="bold",
                 color="white", ha="center", va="center")
        x_rest = x_dec + 0.033 + w_d + 0.012
        arrow(x_rest, x_rest + 0.035, y_tl, ls="--")
        fig.text(x_rest + 0.04, y_tl, "rest of the case:\nnot yet known", fontsize=10.1 * FS, color=INK, va="center", style="italic")

    # --- tres paneles: pregunta -> respuesta ----------------------------------
    COL_W, GAP, X0 = 0.318, 0.011, 0.012
    BOX_Y0, BOX_Y1 = 0.125, 0.715
    cols = [X0 + i * (COL_W + GAP) for i in range(3)]
    LEFT = (0.218, 0.222, 0.240)
    axes = [fig.add_axes([cc + lf, 0.215, COL_W - lf - 0.058, 0.335], zorder=2) for cc, lf in zip(cols, LEFT)]
    for cc in cols:
        box(cc, BOX_Y0, COL_W, BOX_Y1 - BOX_Y0, fc="white", z=-2)

    cate = c["pU"] - c["pT"]
    near = ", near 0.5" if abs(c["r"] - 0.5) < 0.05 else ""
    risk_answer = f"{'Yes' if c['r'] >= 0.5 else 'Not much'}{near}: $r$ = {c['r']:.2f}"
    if c["positive_effect_rule"]:
        effect_answer = f"Yes: $\\hat{{p}}_U$ = {c['pU']:.2f} without, $\\hat{{p}}_T$ = {c['pT']:.2f} with"
    else:
        effect_answer = f"No: $\\hat{{p}}_U$ = {c['pU']:.2f} without, $\\hat{{p}}_T$ = {c['pT']:.2f} with"
    timing_answer = f"{'Intervene' if acts else 'Wait'}: intervene {c['p_act']:.0%}, wait {1 - c['p_act']:.0%}"
    risk_items = [(label(a, v), p, "risk") for a, v, p in c["risk_top"]]
    eff_items = [(label(a, v), p, "effect") for a, v, p, _, _ in c["effect_top"]]
    tk = sorted(c["timing_phi"], key=lambda k: -abs(c["timing_phi"][k]))
    tim_items = []
    for k in tk:
        lab, lvl = state_label(k, st[k], c["prefix_nr"], short=True)
        tim_items.append((lab, c["timing_phi"][k], "case" if lvl == "native" else lvl))
    panels = [
        (axes[0], risk_items, "risk", "1  Is the chance of failure high?", risk_answer, "← lowers    chance of failure    raises →", False),
        (axes[1], eff_items, "effect", "2  Would a second loan offer help?", effect_answer, "← helps less      helps more →", False),
        (axes[2], tim_items, "timing", f"3  Why {'intervene' if acts else 'wait'}, not {'wait' if acts else 'intervene'}?", timing_answer,
         "← reason to wait   reason to intervene →" if acts else "← reason to intervene   reason to wait →", True),
    ]
    for (ax, items, lvl, title, answer, xlabel, tagged), cc in zip(panels, cols):
        phi = np.array([p for _, p, _ in items])
        ax.barh(range(len(items)), phi, color=[POS if v > 0 else NEG for v in phi], height=0.62)
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for yy, v in enumerate(phi):
            ax.text(1.04, yy, signed(v), transform=tr, va="center", ha="left", fontsize=10.5 * FS, color=POS if v > 0 else NEG)
        ax.set_yticks(range(len(items)), [lab for lab, _, _ in items], fontsize=13.0 * FS)
        if tagged:
            for tick, (_, _, lv) in zip(ax.get_yticklabels(), items):
                tick.set_color(LEVEL[lv])
        ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.8)
        if lvl == "timing" and TIMING_LIM:
            ax.set_xlim(-TIMING_LIM, TIMING_LIM)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        box(cc + 0.01, 0.668, 0.012, 0.03, fc=LEVEL[lvl])
        fig.text(cc + 0.028, 0.683, title, fontsize=11.9 * FS, fontweight="bold", color=INK, va="center")
        fig.text(cc + 0.012, 0.638, answer, fontsize=11.2, color=LEVEL[lvl] if lvl != "timing" else (POS if acts else NEG), fontweight="bold",
                 va="center")
        fig.text(cc + COL_W / 2, 0.147, xlabel, fontsize=12.2 * FS, color=INK, ha="center", va="center")
    # revisión: la suma de la antigua Figura 3 (IG sobre Delta Q, o -Delta Q si espera)
    fig.text(cols[2] + 0.012, 0.606, f"$\\Delta Q$ = {signed(c['dq'])};  sum of $\\phi$ = {signed(c['timing_sum'])}", fontsize=9.5, color=INK,
             va="center")
    lx = cols[2] + 0.012
    for i, lv in enumerate(("risk", "effect", "case")):
        xx = lx + i * 0.095
        box(xx, 0.563, 0.01, 0.022, fc=LEVEL[lv])
        fig.text(xx + 0.014, 0.574, {"risk": "from panel 1", "effect": "from panel 2", "case": "from the case"}[lv], fontsize=9.5, color=LEVEL[lv],
                 va="center")
    if PAIR_TITLE and not SHOW_BAND:
        fig.text(0.012, 0.755, f"({'a' if tag == 'A' else 'b'}) Case ${{\it {tag}}}$: {'intervene' if acts else 'wait'}", fontsize=14.5 * FS,
                 fontweight="bold", color=INK, va="center")
    W_IN, H_IN = fig.get_size_inches()
    top = H_IN if SHOW_BAND else (0.79 if PAIR_TITLE else 0.725) * H_IN
    save(fig, f"fig5_three_level_card_{tag}", bbox_inches=Bbox([[0.0, 0.105 * H_IN], [W_IN, top]]))
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    TIMING_LIM = 1.1 * max(abs(v) for t in ("A", "W") for v in d["cards"][t]["timing_phi"].values())
    for tag in (sys.argv[1:] or ["A", "W"]):
        draw(d["cards"][tag], tag)
