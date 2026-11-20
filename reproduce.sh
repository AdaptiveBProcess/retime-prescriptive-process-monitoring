#!/usr/bin/env bash
# Reproduce the paper's results from the committed checkpoints (CPU only, no retraining).
#
#   ./reproduce.sh            set up .venv, run the tests, check the data, run every result script
#   ./reproduce.sh --setup    only create .venv and run the unit tests (no data needed)
#   ./reproduce.sh --restore  put the committed result JSONs back (the *.json at the repo root)
#
# PYTHON=python3.12 ./reproduce.sh picks the interpreter used to create .venv.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
MODE="${1:-all}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ "$MODE" == "--restore" ]]; then
    git checkout -- ':(glob)*.json'
    echo "Committed result JSONs restored."
    exit 0
fi

# 1. environment --------------------------------------------------------------
if [[ ! -x "$PY" ]]; then
    step "Creating .venv"
    "${PYTHON:-python3}" -m venv .venv
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt "pandas<3" pyarrow
else
    step "Using existing .venv ($("$PY" --version))"
fi

step "Unit tests (no data needed, ~30 s)"
"$PY" -m pytest -q test_dual_level.py test_xai_suite.py test_compose.py test_crossfit.py test_risk_model.py

[[ "$MODE" == "--setup" ]] && { echo; echo "Environment ready."; exit 0; }

# 2. data ---------------------------------------------------------------------
step "Checking the data"
REQUIRED=(
    data/retrained_state_bpic2012.csv data/retrained_state_bpic2012_train.csv data/retrained_state_bpic2012_val.csv
    data/retrained_state_bpic2017.csv data/retrained_state_bpic2017_train.csv data/retrained_state_bpic2017_val.csv
    data/retrained_state_sepsis.csv   data/retrained_state_sepsis_train.csv   data/retrained_state_sepsis_val.csv
    data/sepsis_events.parquet
    PeerJ2023/prepared_data/bpic2012/prepared_treatment_outcome_time_to_event_bpic2012.csv
    PeerJ2023/prepared_data/bpic2017/prepared_treatment_outcome_time_to_event_bpic2017.csv
    simbank_resources/data/simbank_time_contact_hq_with_resources.pkl
    simbank_resources/data/simbank_time_contact_hq_with_effect.pkl
)
missing=0
for f in "${REQUIRED[@]}"; do
    [[ -f "$f" ]] || { echo "  missing: $f"; missing=1; }
done
if (( missing )); then
    echo "Put the files above in place (see README.md, 'Put the data in place') and run again."
    echo "Files kept elsewhere can be pointed to with the TIMING_* variables of paths.py."
    exit 1
fi
(cd data && sha256sum -c retrained_states_sha256.txt)

# 3. results ------------------------------------------------------------------
step "fidelity_test.py (Table 4, ~35 s)";              "$PY" fidelity_test.py
step "compute_gain_table.py (gain table, ~10 s)";      "$PY" compute_gain_table.py
step "run_risk_suite.py (risk explanation)";           "$PY" run_risk_suite.py
step "run_effect_suite.py (effect explanation)";       "$PY" run_effect_suite.py
step "run_compose_suite.py (Tables 5-6)";              "$PY" run_compose_suite.py
step "leakage_audit.py (~5 min)"
# On a checkout other than the original, 'agent_trained_on_training_state' fails because it
# compares absolute paths (README, Known issues, item 2); the other checks are the ones to read.
"$PY" leakage_audit.py || echo "  (leakage_audit.py exited non-zero: see the note above)"
step "paper_numbers.py (numbers.tex)";                 "$PY" paper_numbers.py

step "Differences against the committed results"
git diff --stat -- ':(glob)*.json' || true
echo
echo "Only path fields (model, figures) and run times (seconds) are expected to differ,"
echo "plus leakage_audit.json's agent_trained_on_training_state (README, Known issues, item 2)."
echo "Inspect with 'git diff -- <file>.json'; restore with './reproduce.sh --restore'."
