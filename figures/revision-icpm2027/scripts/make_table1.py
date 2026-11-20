"""Table I del running example (Sección 2, Preliminaries): el prefijo común de los dos
casos y una fila de atributos por caso. Escribe el entorno table listo para el paper
(out/table1_running_example.tex) y una vista previa compilada (pdflatex + pdftoppm).
"""

import subprocess
from datetime import datetime

from common import OUT, activity, goal, load

d = load()
A, W = d["cards"]["A"], d["cards"]["W"]
assert [e["activity"] for e in A["trace"]] == [e["activity"] for e in W["trace"]], "los prefijos no son iguales"


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("€", r"\euro{}")


def hm(ts):
    return datetime.fromisoformat(ts).strftime("%H:%M:%S")


def day(ts):
    return datetime.fromisoformat(ts).strftime("%Y-%m-%d")


same_res = all(a["resource"] == w["resource"] for a, w in zip(A["trace"], W["trace"]))
res = (lambda a, w: tex(a["resource"])) if same_res else (lambda a, w: f"{tex(a['resource'])} / {tex(w['resource'])}")
rows = "\n".join(
    f"  {i} & \\texttt{{{tex(a['activity'])}}} & {tex(activity(a['activity']))} & {res(a, w)} \\\\"
    for i, (a, w) in enumerate(zip(A["trace"], W["trace"]), 1))


def attrs(tag, c):
    ca = c["case_attributes"]
    return (f"  {tag} & {tex(c['case_id'].replace('Application_', ''))} & {day(c['trace'][0]['timestamp'])} & "
            f"{tex(ca['ApplicationType'].lower())} & {tex(goal(ca['LoanGoal']))} & {int(float(ca['RequestedAmount'])):,} \\\\")


body = rf"""\begin{{table}}[!htb]
\centering
\caption{{Running example: two BPIC2017 loan applications with the same prefix at their decision point, after the third event; the timing model acts on A and waits on W}}
\label{{tab:running}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}rlll@{{}}}}
\toprule
\# & Activity & In plain terms & {'Resource (A and W)' if same_res else 'Resource (A / W)'} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}

\smallskip
\begin{{tabular}}{{@{{}}lllllr@{{}}}}
\toprule
Case & Application & Date & Type & Loan goal & Requested (\euro{{}}) \\
\midrule
{attrs('A', A)}
{attrs('W', W)}
\bottomrule
\end{{tabular}}

\smallskip
{{\footnotesize Open applications in the log at the decision point: {A['trace'][-1]['open_cases']:,} (A), {W['trace'][-1]['open_cases']:,} (W). All three events of each case share one timestamp ({hm(A['trace'][0]['timestamp'])} for A, {hm(W['trace'][0]['timestamp'])} for W).}}
\end{{table}}
"""
(OUT / "table1_running_example.tex").write_text(body)
print("->", OUT / "table1_running_example.tex")

# vista previa: el mismo entorno, fuera de acmart (solo para mirarlo)
inner = body.replace(r"\begin{table}[!htb]", "").replace(r"\end{table}", "")
preview = rf"""\documentclass{{article}}
\usepackage{{booktabs}}
\usepackage{{eurosym}}
\pagestyle{{empty}}
\begin{{document}}
\begin{{minipage}}{{0.9\textwidth}}
\makeatletter\def\@captype{{table}}\makeatother
{inner}
\end{{minipage}}
\end{{document}}
"""
pv = OUT / "_preview"
pv.mkdir(exist_ok=True)
(pv / "table1_preview.tex").write_text(preview)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "table1_preview.tex"], cwd=pv, check=True, capture_output=True)
subprocess.run(["pdfcrop", "table1_preview.pdf", str(OUT / "table1_running_example_preview.pdf")], cwd=pv, check=True, capture_output=True)
subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(OUT / "table1_running_example_preview.pdf"),
                str(OUT / "table1_running_example_preview")], check=True)
print("->", OUT / "table1_running_example_preview.pdf")
