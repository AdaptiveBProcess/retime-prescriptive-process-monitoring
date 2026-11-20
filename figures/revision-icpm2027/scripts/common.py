"""Lo que comparten las figuras del running example: datos, colores y etiquetas llanas.

Colores: el signo va en naranja/azul (hacia la decisión
o en contra, o phi positivo/negativo) y el nivel de cada feature en el color del texto
de su etiqueta: morado = riesgo, verde azulado = efecto, negro = nativo/caso. Son los
mismos colores que ya usa la Fig. 5 del paper (LEVEL en figures/make_three_level.py).
"""

import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]  # raíz del repo (scripts/ -> revision-icpm2027/ -> figures/ -> repo)
sys.path.insert(0, str(REPO_DIR / "figures"))
from plain import activity, label as _plain_label  # noqa: E402,F401  (plain.py del repo, sin modificarlo)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datos" / "cards_running_example.json"
OUT = HERE.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

POS, NEG, INK, MUTED, LINE = "#c2410c", "#1d4ed8", "#000000", "#6b7280", "#d1d5db"
LEVEL = {"risk": "#7c3aed", "effect": "#0f766e", "case": INK, "native": INK, "timing": INK}
NATIVE = "#8b5e3c"  # café de lo nativo, el mismo de la Figura 1 del paper (nativec)
N_RESOURCES = 3
MINUS = "−"  # signo menos tipográfico
TAGS = {"A": "A", "W": "W"}


def load():
    return json.loads(DATA.read_text())


def signed(v, nd=2):
    return f"{v:+.{nd}f}".replace("-", MINUS)


def goal(v):
    s = str(v).strip()
    return "not specified" if s.lower() in ("not speficied", "not specified") else s.lower()  # el log lo escribe mal


def label(name, value=None):
    """Lectura llana del atributo en el valor que toma aquí."""
    num = value if isinstance(value, (int, float)) else None
    if name == "Activity":
        return "alphabetically last activity (Activity)"
    if name == "LoanGoal" and isinstance(value, str):
        return "no loan goal given" if goal(value) == "not specified" else f"loan goal: {goal(value)}"
    if name == "RequestedAmount" and num is not None:
        return f"RequestedAmount: {num:,.0f}"
    if name.startswith("open_cases") and num is not None:
        agg = {"min": "min", "max": "max", "mean": "mean"}.get(name.rsplit("_", 1)[-1], "")  # agregado sobre el prefijo
        return f"{int(round(num)):,} cases open" + (f" ({agg})" if agg else "")
    if name == "FirstWithdrawalAmount_max" and num == 0:
        return "nothing withdrawn yet"
    if name == "NumberOfTerms_sum" and num == 0:
        return "no terms offered yet (sum)"
    if name == "NumberOfTerms_max" and num == 0:
        return "no terms offered yet (max)"
    if name == "org:resource" and isinstance(value, str):
        return f"handled by {value.replace('_', ' ')}"
    if name.startswith("timesincecasestart"):
        return "time since the start"
    if name.startswith("timesincemidnight"):
        return "time of day"
    if name == "ApplicationType" and isinstance(value, str):
        return f"{value.lower()} application"
    if name.startswith("month") and num is not None:
        import calendar
        return f"applied in {calendar.month_name[int(round(num))]}"
    return _plain_label(name, value)


def state_label(k, v, prefix_nr, short=False):
    """Etiqueta y nivel de una coordenada del estado, con el valor que la política lee.
    short: sin la palabra "rejection" (el panel 3 de la Fig. 5 es estrecho)."""
    wo, wi = ("", "") if short else (" without a second loan offer,", " with a second loan offer,")  # panel 3 is narrow: the symbol says which arm
    return {"Proba_if_Untreated": (f"chance of failure{wo} $\\hat{{p}}_U$ = {v:.2f}", "effect"),
            "Proba_if_Treated": (f"chance of failure{wi} $\\hat{{p}}_T$ = {v:.2f}", "effect"),
            "reliability": (f"reliability {v:.2f}", "risk"),
            "deviation": ("deviation 0 (predicts failure)" if v == 0 else "deviation 1 (predicts success)", "risk"),
            "available_resources": (f"{int(round(v))} of {N_RESOURCES} resources available", "native"),
            "relative_position": (f"relative position {v:.2f}", "native")}[k]


def save(fig, stem, **kw):
    fig.savefig(OUT / f"{stem}.pdf", **kw)
    fig.savefig(OUT / f"{stem}.png", dpi=170, **kw)
    print("->", OUT / f"{stem}.pdf")
