"""Draw each figure of the paper as a cropped PDF in ../figures/ from its TikZ source here.
Labels and citation numbers come from main.aux, given to each figure as its own .aux (copied from a full build of the paper)."""
import re, subprocess, pathlib
here = pathlib.Path(__file__).resolve().parent
out = here.parent / "out"  # in the paper's folder this was ../figures; in the repo the figures go to revision-icpm2027/out/
out.mkdir(exist_ok=True)
PRE = r"""\documentclass[manuscript,nonacm]{acmart}
\usepackage{tikz}\usetikzlibrary{arrows.meta,shapes.geometric}
\colorlet{metaA}{ACMRed}\definecolor{metaB}{RGB}{190,95,0}\definecolor{metaC}{RGB}{40,100,230}\definecolor{metaD}{RGB}{0,128,128}
\colorlet{pcchairs}{magenta!75!black}\definecolor{nometa}{RGB}{90,0,130}
\input{numbers}\input{numbers_ablation}
\newsavebox{\runexbox}
\newenvironment{runexframe}[1][metaB]{\def\runexcol{#1}\setlength{\fboxrule}{0.8pt}\setlength{\fboxsep}{4pt}\noindent\begin{lrbox}{\runexbox}\begin{minipage}{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}}{\end{minipage}\end{lrbox}\fcolorbox{\runexcol}{white}{\usebox{\runexbox}}}
\pagestyle{empty}
\begin{document}\thispagestyle{empty}
"""
for name in ["fig_running_example", "fig1_retime"]:
    src = (here / f"{name}.tex").read_text()
    m = re.search(r"\\begin\{runexframe\}(\[[^\]]*\])?(.*?)\\caption", src, re.S)
    body = m.group(2)
    doc = PRE + r"\begin{runexframe}" + (m.group(1) or "") + body + "\\end{runexframe}\n\\end{document}\n"
    (here / f"_{name}.tex").write_text(doc)
    (here / f"_{name}.aux").write_text((here / "main.aux").read_text())  # labels and citations of the paper
    subprocess.run(["pdflatex", "-interaction=nonstopmode", f"_{name}.tex"], cwd=here, stdout=subprocess.DEVNULL)
    subprocess.run(["pdfcrop", "--margins", "0", f"_{name}.pdf", str(out / f"{name}.pdf")], cwd=here, stdout=subprocess.DEVNULL, check=True)
    log = (here / f"_{name}.log").read_text(errors="ignore")
    for ext in ("tex", "aux", "log", "pdf", "out"):
        (here / f"_{name}.{ext}").unlink(missing_ok=True)
    print(name, "errors:", log.count("\n!"), "undefined refs:", len(re.findall(r"undefined", log)))
