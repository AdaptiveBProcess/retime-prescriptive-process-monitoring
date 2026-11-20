"""Fig. 3 del running example: la explicación de timing de los dos casos lado a lado.
(a) A: Integrated Gradients sobre Delta Q (empuja hacia actuar); (b) W: sobre
-Delta Q (empuja hacia esperar). Misma escala horizontal en los dos paneles, para
que se comparen. La etiqueta de cada coordenada va en el color de su nivel.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import INK, LEVEL, NEG, POS, load, save, signed, state_label  # noqa: E402

d = load()
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#d1d5db"})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 2.9))
lim = max(max(abs(v) for v in d["cards"][t]["timing_phi"].values()) for t in ("A", "W")) * 1.32
for ax, tag in zip(axes, ("A", "W")):
    c = d["cards"][tag]
    acts = c["action"] == "intervene"
    keys = sorted(c["timing_phi"], key=lambda k: -abs(c["timing_phi"][k]))
    phi = np.array([c["timing_phi"][k] for k in keys])
    labs = [state_label(k, c["state"][k], c["prefix_nr"]) for k in keys]
    y = np.arange(len(keys))
    ax.barh(y, phi, color=[POS if v > 0 else NEG for v in phi], height=0.62)
    ax.set_yticks(y, [lab for lab, _ in labs], fontsize=10.5)
    for tick, (_, lvl) in zip(ax.get_yticklabels(), labs):
        tick.set_color(LEVEL[lvl])
    ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(-lim, lim)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    off = 0.03 * lim
    for yi, v in zip(y, phi):
        ax.text(v + (off if v >= 0 else -off), yi, signed(v), va="center", ha="left" if v >= 0 else "right", fontsize=9.5,
                color=POS if v > 0 else NEG)
    what = r"$\phi^{\Delta Q}$: $\leftarrow$ reason to wait     reason to act now $\rightarrow$" if acts else \
        r"$\phi^{-\Delta Q}$: $\leftarrow$ reason to act now     reason to wait $\rightarrow$"
    ax.set_xlabel(what, fontsize=10)
    head = "acts" if acts else "waits"
    ax.set_title(f"({'a' if tag == 'A' else 'b'}) ${{\\it {tag}}}$ {head}: act {c['p_act']:.0%}, wait {1 - c['p_act']:.0%} "
                 f"($\\Delta Q$ = {signed(c['dq'])}; sum of $\\phi$ = {signed(c['timing_sum'])})", fontsize=10.5, loc="left")
# leyenda de niveles, una sola para los dos paneles
for i, (lvl, txt) in enumerate((("risk", "from the risk model"), ("effect", "from the effect model"), ("native", "from the case"))):
    fig.text(0.30 + 0.17 * i, 0.005, "■ " + txt, color=LEVEL[lvl], fontsize=9.5, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "fig3_timing_cards", bbox_inches="tight")
