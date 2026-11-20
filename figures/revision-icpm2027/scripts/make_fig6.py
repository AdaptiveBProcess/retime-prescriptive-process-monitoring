"""Fig. 6 del running example: cómo se compone cada decisión. La explicación de timing
propagada a los atributos del caso por las explicaciones de riesgo y efecto
(compose.propagate), dibujada como flujo: atributos (izquierda) -> lo que lee el
modelo de timing, por explicación (medio) -> la decisión (derecha).

Ancho de cinta = |contribución| en unidades del margen; naranja = hacia la decisión
que tomó el modelo de timing (actuar en A, esperar en W), azul = en contra. Las
etiquetas del medio van en el color de su nivel. Misma maqueta que
figures/make_composition_flow.py del repo (y la versión de figuras-ejemplo), sin las
marcas rojas de revisión.

Escribe fig6_composition_flow_A, fig6_composition_flow_W y fig6_composition_flow_pair
(A arriba, W abajo, con la MISMA escala, para comparar grosores entre los dos casos).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from common import INK, LEVEL, MINUS, MUTED, NATIVE, NEG, POS, label, load, save, signed  # noqa: E402

# revisión: lo nativo en café, como en las Figuras 1 y 3 del paper.
LEVEL = {**LEVEL, "native": NATIVE}

NODE_FC = "#f3f4f6"
MIN_NODE, VSCALE, MIN_RIBBON = 0.055, 0.60, 0.020
KEEP = 3  # atributos con nombre; el resto va a "(other attributes)"
MID = [("risk", "risk model, $r$", "risk"),
       ("effect_T", "chance of failure with\na second loan offer, $\\hat{p}_T$", "effect"),
       ("effect_U", "chance of failure without\na second loan offer, $\\hat{p}_U$", "effect"),
       ("relative_position", "relative position", "native"),
       ("available_resources", "available resources", "native")]


SIDE_BY_SIDE = True  # False: A encima de W, como antes


def collapse(flows, keep=KEEP):
    mag = lambda f: abs(f["risk"]) + abs(f["effect_T"]) + abs(f["effect_U"])  # noqa: E731
    named = [f for f in flows if f["input"] != "(other attributes)"]
    top = sorted(named, key=mag, reverse=True)[:keep]
    names = {f["input"] for f in top}
    rest = [f for f in flows if f["input"] not in names]
    agg = {"input": "(other attributes)", "value": None, **{k: sum(f[k] for f in rest) for k in ("risk", "effect_T", "effect_U")}}
    return top + [agg]


def ribbon(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.32):
    cx = (x0 + x1) / 2
    verts = [(x0, y0), (cx, y0), (cx, y1), (x1, y1), (x1, y1 + h1), (cx, y1 + h1), (cx, y0 + h0), (x0, y0 + h0), (x0, y0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def totals(card):
    t = card["timing_phi"]
    left = collapse(card["flows"])
    mid = {k: sum(abs(f[k]) for f in left) for k in ("risk", "effect_T", "effect_U")}
    mid.update(relative_position=abs(t["relative_position"]), available_resources=abs(t["available_resources"]))
    return sum(mid.values())


def flow(ax, card, scale, title=None):
    acts = card["action"] == "intervene"
    t = card["timing_phi"]
    labelled = {"Activity", card["flows"][0]["input"]}  # Activity y el atributo de arriba por |total|: los caminos que cita el texto del paper
    flows = collapse(card["flows"])
    left = [(f["input"], f.get("value"), {k: f[k] for k in ("risk", "effect_T", "effect_U")}) for f in flows]
    mid_in = {"risk": t["reliability"] + t["deviation"], "effect_T": t["Proba_if_Treated"], "effect_U": t["Proba_if_Untreated"],
              "relative_position": t["relative_position"], "available_resources": t["available_resources"]}
    mid_abs = {"risk": abs(t["reliability"]) + abs(t["deviation"]), "effect_T": abs(t["Proba_if_Treated"]),
               "effect_U": abs(t["Proba_if_Untreated"]), "relative_position": abs(t["relative_position"]),
               "available_resources": abs(t["available_resources"])}
    mid_h = {k: sum(abs(c[k]) for _, _, c in left) if k in ("risk", "effect_T", "effect_U") else mid_abs[k] for k, _, _ in MID}
    left_h = [sum(abs(v) for v in c.values()) for _, _, c in left]

    ax.set_xlim(0, 10); ax.axis("off")
    TOPY = 0.9
    X0, X1, X2, W = 1.3, 5.15, 9.0, 0.22
    gap_l, gap_m = 0.055, 0.045
    hl = [h * scale for h in left_h]
    y = TOPY - (TOPY - (sum(hl) + gap_l * (len(hl) - 1))) / 2
    left_pos = []
    for h in hl:
        left_pos.append((y - h, h)); y -= h + gap_l
    hm = [max(mid_h[k] * scale, MIN_NODE) for k, _, _ in MID]
    y = TOPY - (TOPY - (sum(hm) + gap_m * (len(hm) - 1))) / 2
    mid_pos = {}
    for (k, _, _), h in zip(MID, hm):
        mid_pos[k] = (y - h, h); y -= h + gap_m
    rib_h = {k: max(mid_abs[k] * scale, MIN_RIBBON) if mid_abs[k] > 0 else 0.0 for k, _, _ in MID}
    right_h = sum(rib_h.values())
    right_pos = (TOPY / 2 - right_h / 2, right_h)

    cursor_l = [p[0] + p[1] for p in left_pos]
    cursor_m = {k: mid_pos[k][0] + (mid_pos[k][1] + mid_h[k] * scale) / 2 for k in mid_pos}
    for i, (name, _, c) in enumerate(left):
        for k in ("risk", "effect_T", "effect_U"):
            h = abs(c[k]) * scale
            if h <= 0:
                continue
            color = POS if c[k] > 0 else NEG
            ribbon(ax, X0 + W, cursor_l[i] - h, h, X1, cursor_m[k] - h, h, color)
            if name in labelled and abs(c[k]) >= 0.05:
                ax.text(X1 - 0.06, cursor_m[k] - h / 2, signed(c[k]), ha="right", va="center", fontsize=10.0, color=color,
                        fontweight="bold", zorder=4, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8))
            cursor_l[i] -= h; cursor_m[k] -= h
    cursor_r = right_pos[0] + right_pos[1]
    for k, _, _ in MID:
        h = rib_h[k]
        if h <= 0:
            continue
        y0 = mid_pos[k][0] + (mid_pos[k][1] - h) / 2
        ribbon(ax, X1 + W, y0, h, X2, cursor_r - h, h, POS if mid_in[k] > 0 else NEG, alpha=0.32 if h > 0.06 else 0.5)
        cursor_r -= h

    MINSEP_L = 0.165
    lys = []
    for y0, h in left_pos:
        cc = y0 + h / 2
        lys.append(cc if not lys else min(cc, lys[-1] - MINSEP_L))
    for (name, val, c), (y0, h), ly in zip(left, left_pos, lys):
        ax.add_patch(plt.Rectangle((X0, y0), W, h, fc=NODE_FC, ec=INK, lw=0.6, zorder=2))
        lab = label(name, val) if name != "(other attributes)" else "(other case attributes)"
        if name in labelled:
            tot = sum(abs(v) for v in c.values())
            if tot:
                lab += f"\ncancellation $c$ = {1 - abs(sum(c.values())) / tot:.2f}"
        if abs(ly - (y0 + h / 2)) > 1e-6:
            ax.plot([X0 - 0.06, X0], [ly, y0 + h / 2], color=MUTED, lw=0.5, zorder=2)
        ax.text(X0 - 0.08, ly, lab, ha="right", va="center", fontsize=11.2, color=INK)
    for k, lab, lvl in MID:
        y0, h = mid_pos[k]
        ax.add_patch(plt.Rectangle((X1, y0), W, h, fc=NODE_FC, ec=INK, lw=0.6, zorder=2))
        ax.text(X1 + W + 0.08, y0 + h / 2, lab, ha="left", va="center", fontsize=11.2, color=LEVEL[lvl], zorder=3,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    y0, h = right_pos
    ax.add_patch(plt.Rectangle((X2, y0), W, h, fc=NODE_FC, ec=INK, lw=0.8, zorder=2))
    p = card["p_act"]
    mlabel = f"intervene\n(intervene {p:.0%},\nwait {1 - p:.0%})" if acts else f"wait\n(wait {1 - p:.0%},\nintervene {p:.0%})"
    ax.text(X2 + W + 0.08, TOPY / 2, mlabel, ha="left", va="center", fontsize=11.2, color=POS, fontweight="bold")
    ax.text(X0 + W / 2, 1.17, "case attributes", ha="center", va="center", fontsize=11.2, color=INK)
    ax.text(X1 + W / 2, 1.17, "what the timing model reads,\nby explanation" if SIDE_BY_SIDE else "what the timing model reads, by explanation", ha="center", va="center", linespacing=1.1, fontsize=11.2, color=INK)
    ax.text(X2 + W / 2, 1.17, "decision", ha="center", va="center", fontsize=11.2, color=INK)
    if title:
        ax.text(-0.9, 1.36, title, ha="left", va="center", fontsize=11.8, fontweight="bold", color=INK)
    y_low = min([p_[0] for p_ in mid_pos.values()] + [p_[0] for p_ in left_pos] + [right_pos[0]])
    ax.set_ylim(min(0.07, y_low - 0.05), 1.45 if title else 1.26)


def title(tag, c):
    head = "intervene" if c["action"] == "intervene" else "wait"
    to = "intervening" if c["action"] == "intervene" else "waiting"
    return f"({'a' if tag == 'A' else 'b'}) Case ${{\\it {tag}}}$: {head}  (orange: toward {to}, blue: against it)"


if __name__ == "__main__":
    d = load()
    cards = d["cards"]
    for tag in ("A", "W"):
        fig, ax = plt.subplots(figsize=(11.0, 2.62))
        flow(ax, cards[tag], VSCALE / totals(cards[tag]))
        save(fig, f"fig6_composition_flow_{tag}", bbox_inches="tight")
        plt.close(fig)
    common = 0.85 / max(totals(cards["A"]), totals(cards["W"]))  # misma escala: los grosores se comparan entre casos
    if SIDE_BY_SIDE:  # revisión: A y W lado a lado, a lo ancho de la página
        fig, axes = plt.subplots(1, 2, figsize=(19.0, 3.9))
        for ax, tag in zip(axes, ("A", "W")):
            flow(ax, cards[tag], common, title(tag, cards[tag]))
        fig.subplots_adjust(wspace=0.42)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.6))
        for ax, tag in zip(axes, ("A", "W")):
            flow(ax, cards[tag], common, title(tag, cards[tag]))
        fig.subplots_adjust(hspace=0.25)
    save(fig, "fig6_composition_flow_pair", bbox_inches="tight")
    print("escala común: 1 unidad del margen =", round(common, 4), "de alto (A total", round(totals(cards["A"]), 3), ", W total",
          round(totals(cards["W"]), 3), ")")
