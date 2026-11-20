r"""LaTeX macros for the paper's running example (Table 1): the prefix of the
act card's case at its decision point, read from the same prepared BPIC2017
log the risk and effect models are trained on.

Also the cancellation ratio c_j = 1 - |sum of paths| / sum |paths| of the
card's top composed attribute (compose_results.json, ``readings.rebuilt``,
the same reading paper_numbers.py takes \CARDACTFLOWTOP* from).

The case and the decision point come from risk_results.json
(``cards.paper_card_act``), so the table always shows the case that the
"In our example" sentences and Figure 3 describe. The prefix is ordered as
figures/make_three_level.py orders it (timestamp, then activity).

Standard library only, so it also runs where the full pipeline's
dependencies are not installed:

    python running_example_numbers.py <paper>/numbers.tex

rewrites only the block between the RUNNING-EXAMPLE markers of an existing
numbers.tex. paper_numbers.py calls :func:`macros` and writes the same block.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import paths

LOG = "BPIC2017"
CASE_ATTRS = ("ApplicationType", "LoanGoal", "RequestedAmount")  # static case attributes shown in the caption
BEGIN = "%% BEGIN RUNNING-EXAMPLE (running_example_numbers.py)"
END = "%% END RUNNING-EXAMPLE"


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _card() -> tuple[str, int]:
    risk = {x["log"]: x for x in json.loads(paths.RISK_JSON.read_text())["results"]}[LOG]
    c = risk["cards"]["paper_card_act"]
    return c["case_id"], int(c["prefix_nr"])


def _prefix(case_id: str, prefix_nr: int) -> list[dict]:
    src = paths.PREPARED_LOGS / "bpic2017/prepared_treatment_outcome_time_to_event_bpic2017.csv"
    rows = []
    with src.open(newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["Case ID"] == case_id:
                rows.append(r)
            elif rows:  # a case's events are contiguous in the prepared log
                break
    if len(rows) < prefix_nr:
        raise ValueError(f"{case_id}: {len(rows)} events in {src}, need {prefix_nr}")
    rows.sort(key=lambda r: (datetime.fromisoformat(r["time:timestamp"]), r["Activity"]))
    return rows[:prefix_nr]


def _flow_cancellation() -> float:
    comp = {x["log"]: x for x in json.loads(paths.COMPOSE_JSON.read_text())["results"]}[LOG]
    f0 = comp["readings"]["rebuilt"]["cards"]["paper_card_act"]["flows"][0]
    paths_ = [f0["risk"], f0["effect_T"], f0["effect_U"]]
    return 1 - abs(sum(paths_)) / sum(abs(v) for v in paths_)


def _hand_risk_path() -> dict:
    """The numbers of the risk path of the top attribute, so the reader can
    redo the rescaling rule by hand: the timing explanation's risk mass
    (reliability + deviation) times the attribute's share of the risk
    explanation, phi^r_j / sum phi^r, where sum phi^r = logit r - E[logit r].
    Raises if the stored path does not match (fallback used, or another card)."""
    import math
    risk = {x["log"]: x for x in json.loads(paths.RISK_JSON.read_text())["results"]}[LOG]
    rc = risk["cards"]["paper_card_act"]
    comp = {x["log"]: x for x in json.loads(paths.COMPOSE_JSON.read_text())["results"]}[LOG]
    cc = comp["readings"]["rebuilt"]["cards"]["paper_card_act"]
    f0 = cc["flows"][0]
    top = {a: v for a, _, v in rc["risk_top"]}
    logit_r = math.log(rc["r"] / (1 - rc["r"]))
    base = risk["risk"]["expected_value_logodds"]
    mass = cc["timing_phi"]["reliability"] + cc["timing_phi"]["deviation"]
    path = mass * top[f0["input"]] / (logit_r - base)
    if abs(path - f0["risk"]) > 1e-6:
        raise ValueError(f"hand risk path {path} != stored {f0['risk']}")
    return {"logit_r": logit_r, "base": base, "sum": logit_r - base, "phi": top[f0["input"]], "mass": mass, "path": path}


def macros() -> list[str]:
    case_id, prefix_nr = _card()
    ev = _prefix(case_id, prefix_nr)
    table = "\n".join(
        f"{i} & \\texttt{{{_tex(r['Activity'])}}} & {_tex(r['org:resource'])} & "
        f"{datetime.fromisoformat(r['time:timestamp']).strftime('%Y-%m-%d %H:%M:%S')} & {int(float(r['open_cases']))} \\\\"
        for i, r in enumerate(ev, 1))
    first = ev[0]
    h = _hand_risk_path()
    for a in CASE_ATTRS:  # static attributes: one value per case
        if any(r[a] != first[a] for r in ev):
            raise ValueError(f"{case_id}: {a} varies within the prefix")
    return [
        BEGIN,
        f"\\newcommand{{\\REPREFIXROWS}}{{{table}}}",
        f"\\newcommand{{\\REAPPLICATIONTYPE}}{{{_tex(first['ApplicationType'])}}}",
        f"\\newcommand{{\\RELOANGOAL}}{{{_tex(first['LoanGoal'])}}}",
        f"\\newcommand{{\\REREQUESTEDAMOUNT}}{{{int(float(first['RequestedAmount'])):,}}}",
        *[f"\\newcommand{{\\{k}}}{{{v:.2f}}}" for k, v in (
            ("RELOGITR", h["logit_r"]), ("RERISKBASE", h["base"]), ("REPHIRSUM", h["sum"]),
            ("REPHIRTOP", h["phi"]), ("RETIMINGRISK", h["mass"]), ("REHANDRISKPATH", h["path"]))],
        f"\\newcommand{{\\REFLOWTOPCANC}}{{{100 * _flow_cancellation():.0f}\\%}}",  # percent, like CANCMEDIAN and the other cancellation ratios
        END,
    ]


def rewrite(numbers_tex: Path) -> None:
    text = numbers_tex.read_text()
    block = "\n".join(macros())
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        text = head + block + tail
    else:
        text = text.rstrip("\n") + "\n" + block + "\n"
    numbers_tex.write_text(text)
    print(f"running-example macros -> {numbers_tex}")


if __name__ == "__main__":
    rewrite(Path(sys.argv[1]) if len(sys.argv) > 1 else paths.PAPER_FIGURES.parent / "numbers.tex")
