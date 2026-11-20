# Figuras del running example (revisión ICPM 2027)

Figuras con datos del paper revisado: Fig. 3 (timing cards A/W), Fig. 5 (tarjeta de tres niveles de A)
y Fig. 6 (flujo de composición del par). Las Figuras 1 (RETIME) y 2 (running example) están en TikZ,
dentro del propio paper (`fig1_retime.tex`, `fig_running_example.tex`).

- `datos/cards_running_example.json`: valores por decisión de A (Application_381087716) y
  W (Application_519115732), calculados con los modelos guardados del repo (sin reentrenar).
  Se regenera con `scripts/build_cards.py`.
- `scripts/make_fig3.py`, `make_fig5.py`, `make_fig6.py`: escriben los PDF/PNG en `out/`.
  `common.py` tiene colores, etiquetas y la cursiva de A/W; usa `figures/plain.py` del repo.
- `scripts/make_fig1.py`, `make_table1.py`: borradores del running example (en el paper se editaron a mano).
- `tests_pair/tests_pair.py <salida.json>`: pruebas por decisión de A y W (borrado, patching, etc.);
  `tests_pair.json` es la salida usada en el texto.

- `tikz/`: fuentes TikZ de las Figuras 1 (running example, `fig_running_example.tex`) y 2 (RETIME,
  `fig1_retime.tex`) del paper. `python tikz/build_figures.py` las compila (pdflatex + pdfcrop) y escribe
  `out/fig_running_example.pdf` y `out/fig1_retime.pdf`; toma las referencias y citas de `tikz/main.aux`,
  copiado de una compilación completa del paper, y lee `numbers.tex` y `numbers_ablation.tex`.

Entorno: Python 3.11.13 (pyenv) con los paquetes del repo más catboost, xgboost, shap y captum.
Con matplotlib 3.10.8 las Figuras 3 y 4 salen iguales, píxel a píxel, a las del paper; otras versiones
de matplotlib cambian levemente el dibujo, no los valores.

    python scripts/make_fig3.py   # idem fig5, fig6

Después copiar `out/fig3_timing_cards.pdf`, `out/fig5_three_level_card_A.pdf` y
`out/fig6_composition_flow_pair.pdf` a la carpeta `figures/` del paper.

Nota (revisión posterior): la Fig. 3 (timing cards) se quitó del paper; su contenido está en el panel 3 de
`fig5_three_level_card_A/W.pdf`, que ahora es la Fig. 3 del paper (A y W lado a lado, misma escala en el panel 3,
con ΔQ y la suma de φ). `make_fig3.py` se conserva. La antigua Fig. 6 (flujo) es ahora la Fig. 4, con A y W lado a lado.

Nota (color): en las Figs. 3 y 4 del paper (make_fig5.py, make_fig6.py) lo nativo va en café `NATIVE = "#8b5e3c"` (common.py), el mismo de la Fig. 1 (`nativec`).
