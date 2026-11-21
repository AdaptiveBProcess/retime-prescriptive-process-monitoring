# retime-prescriptive-process-monitoring

Supporting code for the paper *"How Multiple Explanations Behind a Decision
Compose in Prescriptive Process Monitoring"* (RETIME: Risk, Effect and TIming
Multiple Explanations; ICPM 2027).

A PsPM decision comes from a chain of three models: a *risk model* (CatBoost)
that predicts the chance of failure, an *effect model* (one XGBoost per arm)
that estimates the chance of failure with and without an intervention, and a
*timing model* (a PPO policy, Shoush & Dumas's design) that reads both outputs
in a six-feature state and decides whether to act now or wait. RETIME has four
steps: (i) the *risk and effect explanations* (TreeSHAP on the log-odds of each
model), (ii) the *timing explanation* (`φ^ΔQ`: Integrated Gradients on the
margin ΔQ = log π(act|s) − log π(wait|s), from the pool average, sign-flipped
where the timing model waits), (iii) the *composition*: `compose.py` traces the
timing explanation back to the case attributes with the rescaling rule of
DeepLIFT/DeepSHAP (as Chen et al. 2022 do for a series of models), keeping
each path and measuring each explanation's share and a per-attribute
cancellation ratio (after Kramár et al.'s AtP*, 2024), and (iv) the
*verification* on the whole chain F(x, n) = ΔQ(s(x, n)): a deletion test
(Samek et al. 2017), a sampled direct explanation, and path patching
(Goldowsky-Dill et al. 2023). Risk-vs-effect agreement is read with the
rank and sign metrics of Krishna et al.'s disagreement-problem study.

**Contents:** [Quick start](#quick-start) ·
[What each step needs](#what-each-step-needs) ·
[Known issues on a fresh clone](#known-issues-on-a-fresh-clone) ·
[Paper ↔ code](#paper--code) · [The explained policy](#the-explained-policy) ·
[Leakage](#leakage) · [Architecture and conventions](#architecture-and-conventions) ·
[Layout](#layout) · [All commands](#all-commands) · [Data](#data) ·
[Retraining](#retraining-the-checkpoints) ·
[Findings about the released data](#findings-about-the-released-data-that-the-code-depends-on) ·
[Literature XAI suite](#literature-explanation-methods-and-metrics-run_xai_suitepy)

---

## Quick start

Everything runs on CPU. No GPU is needed and no model has to be retrained: the
trained checkpoints are committed, and every script reads them.

**Tested with** Python 3.14.8 on Linux. The package versions were torch 2.15.0
(CPU), stable-baselines3 2.9.0, pandas 2.3.3, numpy 2.5.5, catboost 1.2.10,
xgboost 3.4.1 and shap 0.54.1. Any Python from 3.10 to 3.14 should work. You
need about 3 GB of disk for the environment.

### One command

Once the data is in place (step 2 below), this does steps 1 to 3:

```bash
git clone https://github.com/andresmosqueraw/retime-prescriptive-process-monitoring.git
cd retime-prescriptive-process-monitoring
./reproduce.sh
```

`reproduce.sh` creates `.venv` if it does not exist and runs the unit tests.
It then checks that every input file is present and matches its SHA-256 sum,
runs every result script, and ends with `git diff --stat` against the
committed JSONs. Other modes:

| Command | What it does |
|---|---|
| `./reproduce.sh --setup` | only create `.venv` and run the 47 unit tests (no data needed) |
| `./reproduce.sh --restore` | put the committed result JSONs back |
| `PYTHON=python3.12 ./reproduce.sh` | create `.venv` with a specific interpreter |

The script needs bash, so on Windows run it from WSL or Git Bash. The
manual steps below do the same thing one command at a time.

### 1. Clone and create the environment

```bash
git clone https://github.com/andresmosqueraw/retime-prescriptive-process-monitoring.git
cd retime-prescriptive-process-monitoring
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt "pandas<3" pyarrow
```

The two extra packages on the last line are required (see
[Known issues](#known-issues-on-a-fresh-clone)). The `--extra-index-url`
installs the CPU build of torch, which is about 200 MB instead of several GB.
On Windows, replace `.venv/bin/` with `.venv\Scripts\` in every command.

Check the environment (47 tests, about 30 s; none of them needs data):

```bash
.venv/bin/python -m pytest test_dual_level.py test_xai_suite.py test_compose.py test_crossfit.py test_risk_model.py -q
```

### 2. Put the data in place

The logs are third-party data and are not in the repository. Copy them to the
locations below, or point to them with the environment variables in
[Data](#data):

```
data/retrained_state_{bpic2012,bpic2017,sepsis}{,_train,_val}.csv    # the timing model's state (9 files)
data/ready_to_use_adaptive_bpic2012.csv                               # Shoush & Dumas's shipped RL CSVs
data/ready_to_use_adaptive_bpic2017.csv
data/sepsis_events.parquet                                            # Sepsis Cases event log, one row per event
PeerJ2023/prepared_data/bpic2012/prepared_treatment_outcome_time_to_event_bpic2012.csv
PeerJ2023/prepared_data/bpic2017/prepared_treatment_outcome_time_to_event_bpic2017.csv
simbank_resources/data/simbank_time_contact_hq_with_resources.pkl
simbank_resources/data/simbank_time_contact_hq_with_effect.pkl
```

Check that the nine state CSVs are the ones the results were computed on (run
it from inside `data/`, because the checksum file lists bare file names):

```bash
(cd data && sha256sum -c retrained_states_sha256.txt)
```

### 3. Reproduce the results

Copy the whole block into a terminal. Each script overwrites its committed
result JSON, so the closing `git diff --stat` shows anything that changed:

```bash
set -e
PY=.venv/bin/python
$PY fidelity_test.py          # timing deletion test + level shares (Table 4)       ~35 s
$PY compute_gain_table.py     # gain per decision point on the test cases (§4.1)   ~10 s
$PY run_risk_suite.py         # risk explanation (TreeSHAP)                         minutes
$PY run_effect_suite.py       # effect explanation (TreeSHAP per arm)               ~5 min
$PY run_compose_suite.py      # composition study (§5.2-5.3, Tables 5-6)           minutes
$PY leakage_audit.py || true  # future-information audit                          ~5 min (see Known issues)
$PY paper_numbers.py          # every number in the paper -> figures/numbers.tex
git diff --stat
```

What to expect:

- `fidelity_results.json` and `gain_table_results.json` reproduce bit for bit.
  The only field that changes is `"model"`, the absolute path of the
  checkpoint on your machine.
- In the other JSONs, expect differences only in path fields (`model`,
  `figures`) and run times (`seconds`). The exception is `leakage_audit.json`,
  where `agent_trained_on_training_state` turns false on a new checkout
  (Known issues, item 2). Running a suite with `--skip-plots` empties its `figures` list.
- `paper_numbers.py` writes its LaTeX macros to `numbers.tex`
  next to the figures folder. To write straight into a manuscript, run
  `TIMING_PAPER_FIGURES=<paper>/figures .venv/bin/python paper_numbers.py`,
  which writes `<paper>/numbers.tex`.
- The figures go to `figures/out/` (git-ignored). The revised paper's figures
  are built separately: see [Paper ↔ code](#paper--code).

To restore the committed results afterwards, run
`git checkout -- ':(glob)*.json'` (the result JSONs at the repository root),
or `./reproduce.sh --restore`.

---

## What each step needs

| Step | Reads | Time |
|---|---|---|
| unit tests | nothing | ~30 s |
| `fidelity_test.py` | committed checkpoints, `data/retrained_state_*.csv` (test), the SimBank effect pkl, `data/retrained_state_sepsis.csv` | ~35 s |
| `compute_gain_table.py` | the same + the `_val.csv` states | ~10 s |
| `run_risk_suite.py`, `run_effect_suite.py`, `run_compose_suite.py` | the same + `PeerJ2023/prepared_data/`, `data/sepsis_events.parquet`, the SimBank resources pkl | minutes each |
| `leakage_audit.py` | prepared logs + all nine state CSVs | ~5 min |
| `paper_numbers.py` | the committed JSONs, `data/retrained_state_bpic2017.csv`, the BPIC2017 prepared log (running example) | seconds |
| `run_xai_suite.py` (not in the paper) | the state CSVs | ~10 min |
| `--variant cate` / `released` of any script | Shoush & Dumas's shipped CSVs `data/ready_to_use_adaptive_*.csv` | as above |

Most scripts accept `--logs` to run a subset, and the suites accept
`--quick N` for a smoke run on N pool rows. Example:
`.venv/bin/python run_xai_suite.py --logs BPIC2017 --quick 60 --n-lime 30 --stability-calls 3 --out /tmp/x.json`.

## Known issues on a fresh clone

These come from the code as committed. Each one has a workaround that needs no
code change:

1. **`requirements.txt` misses two constraints.** It does not list `pyarrow`
   (needed to read `data/sepsis_events.parquet`), and it allows pandas 3,
   which breaks `leakage_audit.py`
   (`ValueError: could not convert string to float: 'True'`). The install
   command in [Quick start](#1-clone-and-create-the-environment) adds both
   (`"pandas<3" pyarrow`).
2. **`leakage_audit.py` reports one false FAIL outside the original
   checkout.** The check `agent_trained_on_training_state` compares the
   absolute `csv_path` stored in the checkpoint manifest with the local path
   of `data/retrained_state_<log>_train.csv`. On any other machine the paths
   differ, so the check fails, and the script exits with code 1. The file is
   the same: its name and its SHA-256 match. Every other check is the one to
   read.
3. **The scripts under `figures/revision-icpm2027/` have a hard-coded
   absolute path.** `scripts/build_cards.py`, `scripts/common.py` and
   `tests_pair/tests_pair.py` set `REPO` / `REPO_DIR` to the author's
   checkout. Change that line to the path of your clone before you run them.
4. **`sepsis_resources/train_ppo_sepsis.py` defaults to the test state.**
   Without `--csv` it trains on `data/retrained_state_sepsis.csv`, the test
   split. To retrain under the paper's protocol, always pass
   `--csv data/retrained_state_sepsis_train.csv`, as in
   [Retraining](#retraining-the-checkpoints). The same caution applies to
   `evaluate_checkpoints.py`, which scores checkpoints on the test state and
   is only for the legacy sweep.
5. **The SimBank checkpoint names lose their `.zip`.** SB3 drops the `.zip`
   suffix when the output name contains a dot (`ppo_bpic2017_ent0.3`).
   `select_policy.py` already handles this. If you load such a file by hand,
   pass the exact file name.

---

## Paper ↔ code

| Paper (revised manuscript) | Code |
|---|---|
| Fig. 1 and the running example, applications A (`Application_381087716`) and W (`Application_519115732`) | `figures/revision-icpm2027/` (`datos/cards_running_example.json`, `scripts/build_cards.py`, `tests_pair/`); Figs. 1–2 are TikZ in the paper |
| Figs. 3 (A and W explained alone) and 4 (composed explanations) | `figures/revision-icpm2027/scripts/make_fig5.py`, `make_fig6.py` |
| §4 logs, Sepsis overlap, leakage | `risk_model.py`, `leakage_audit.py` |
| §4.1 chain of models, risk AUC 0.69, arm AUCs 0.62 / 0.72 (BPIC2017) | `risk_model.py`, `effect_model.py`, `build_retrained_state.py`, `crossfit.py` (5 folds), `select_policy.py` (10 candidates) |
| §4.1 gain 47.2 / 49.5 per decision point | `compute_gain_table.py` → `gain_table_results.json` |
| §4.2 pool (one prefix per test case) | `pools.py` |
| §5.1, Table 4 (shares, timing deletion test) | `fidelity_test.py` → `fidelity_results.json` |
| §5.2–5.3, Tables 5–6 (paths, cancellation, assumptions, direct explanation, path patching) | `compose.py`, `run_compose_suite.py` → `compose_results.json` |
| Every number in the text | `paper_numbers.py` → `numbers.tex`; `running_example_numbers.py` for the running example |

Some names in the code predate the paper's wording: *timing justification* =
timing explanation, *prefix attributes* = case attributes, *policy* / *agent* =
timing model, *channel* = path, *baseline* / *background* = reference,
*fallback* = safeguard, *treatable* = the positive-effect rule.
`compose.py` also cross-tabulates the decisions by (at risk) × (treatable) ×
(acts), which the paper does not report.

## The explained policy

The paper's timing model (`pools.DEFAULT_VARIANT = "cate_retrained"`,
`models/variants/ppo_<log>_cate_retrained.zip`) observes the released four
features plus the effect model's two outputs (`Proba_if_Treated` = p_T,
`Proba_if_Untreated` = p_U), and is trained one case per episode with
`--reward-scale 0.01 --random-resources` on the *coherent* state, under a
train / validation / test protocol: `build_retrained_state.py` scores the
training split out of fold (`crossfit.py`, 5 folds by case) and the validation
and test splits with the final retrained risk model (`risk_model.py`) and
effect model (`effect_model.py`), into RL CSVs of Shoush & Dumas's format
(`data/retrained_state_<log>_train.csv`, `_val.csv`, `.csv`), so the risk and
effect explanations describe exactly the numbers the timing model reads. The
agents train on the training state only (sweep in `models/variants/sweep_oof/`:
ent_coef 0.1 / 0.3 / 0.5, plus 0.7 / 1.0 on the BPIC logs, × 300k / 600k
steps); `select_policy.py` picks each log's checkpoint by mean reward per
decision point on validation (`policy_selection.json`: BPIC2017 `ent1.0`,
BPIC2012 `ent0.5`, both 600k steps), and the test split is used once, at the
end. Every script defaults to this variant.
`--variant cate` is the same recipe on the shipped CSVs (`*_cate.json`):
there the retrained lower levels explain *other* numbers than the state
carries (corr 0.02 on r and 0.07 on p_T on BPIC2017), and rebuilding the
state from them flips 26 % / 39 % of the pool's decisions on BPIC2012 /
BPIC2017 (`compose_results_cate.json`), which is why the coherent variant is
the paper's. `--variant released` evaluates the earlier four-feature design
(`*_released.json`). SimBank keeps its single checkpoint, trained on the
retrained estimator's features (`simbank_resources/add_effect_features.py`).

Per decision point on the test cases (`gain_table_results.json`, corrected
reward) the paper's timing model earns 49.5 / 47.2 / 8.0 (BPIC2012 / BPIC2017 /
Sepsis) against -15.5 / 20.2 / -3.8 for always waiting and -9.5 / -45.2 / -21.2
for always intervening (oracle 64.8 / 55.7 / 8.1). It intervenes on 50 % / 23 %
of the BPIC2012 / BPIC2017 decision points, with precision 0.85 / 0.81 and
recall 0.92 / 0.91 against the rows where the positive-effect rule holds. On
validation, where it was chosen, it earns 50.7 / 47.8.

## Leakage

The released configuration lets the future of a case into its prefix; the
pipeline here removes it and `leakage_audit.py` checks it (all three logs
pass, `leakage_audit.json`): no whole-case `NumberOfOffers` (it also defined
the released treatment), no BPIC2017 `CreditScore` (filled in only on the
offer the applicant ends up accepting: P(deviant | seen) = 0.00006) with the
offer's `Accepted`/`Selected`, no `time_to_event_m`; decision points end
before the outcome shows (incl. O_Accepted) and before the treatment (a
second offer; IV Antibiotics), models are trained on those prefixes only,
static attributes take the value known so far, and the state's
`relative_position` divides by a fixed training-split horizon instead of the
case's own length. Without these the BPIC2017 risk model reached AUC 0.94,
almost all of it on CreditScore; without them it reaches 0.69. The reward's
sign is also corrected (`compose.desired_outcome`). Pre-fix artefacts are in
`archive/pre_reward_fix/` (git-ignored).

## Architecture and conventions

- **Paths.** `paths.py` is the single source of every input and output path,
  each with a `TIMING_*` environment override. `data/*.csv` and
  `data/loan_log_*` are third-party logs and are git-ignored.
- **Pools.** `pools.py` defines the state pools every script shares: the
  evaluation pool (`bpic_pool` / `simbank_pool` / `sepsis_pool`, seed 123,
  one prefix per case of *every* test case; `N_CASES` is set above every
  split's size) and the full decision-point pools of the gain table. The BPIC
  logs record no live capacity, so the evaluation pool *cycles*
  `available_resources` through 0..3 while the full pool holds it at 3; the
  coherent BPIC policies are trained with `--random-resources` (initial free
  resources uniform in 0..3 per episode), so that range is in distribution.
  SimBank's column is real (synthetic congestion from `build_resources.py`)
  and SimBank is evaluated on every event (`rows="all"`).
- **Timing level.** `dual_level.py` holds the differentiable heads over the
  SB3 `ActorCriticPolicy` (`MarginHead`, `WaitMarginHead`, `CriticHead`,
  branch-wise logit heads), Integrated Gradients (128-step straight path,
  reference = pool mean) and `deletion_test` (guided / random ×20 /
  anti-guided masking to the reference, paired SE). Wait states are
  `margin <= 0`; the intervene action index is 1.
- **Risk level.** `risk_model.py` retrains Shoush & Dumas's CatBoost
  (ensemble size 1) from the prepared logs, with their split and encoding
  reimplemented incrementally in `encode_prefixes` (checked against their
  materialised pipeline in `test_risk_model.py`; `foreign/common_files/` is
  their code, one path patched). `deviation = 1 − 1[r > 0.5]`, where class 1
  is the *bad* outcome, so `deviation = 1` means "predicted to end well";
  `reliability = deviation` on BPIC2012 and `max(r, 1 − r)` on BPIC2017.
- **Effect level.** `effect_model.py` retrains CausalLift's two-model
  estimator (per-arm `XGBClassifier(max_depth=3, n_estimators=100, lr=0.1)`
  with IPW weights from a `LogisticRegression` propensity). φ^CATE =
  φ^{p_U} − φ^{p_T} (toward a larger benefit). On SimBank the estimator is
  fit on the four events both arms share and the untreated arm is a
  `ConstantArm` (skipping always cancels).
- **Composition.** `compose.py`: `rebuild_state` rebuilds the state from the
  retrained models; `propagate` splits each `φ^ΔQ_i` among the prefix
  attributes in proportion to the lower-level attribution, keeping the
  channels (`risk`, `effect_T`, `effect_U`, `native`) apart and conserving
  completeness (a lower-level attribution whose sum is below
  `FALLBACK_RATIO = 0.05` of its |φ| mass is split by |φ|); `cancellation`,
  `well_defined`, `baseline_alignment`, `sign_groups`; `anchor_to_direct`
  (direct ranking, propagated channel split); `EndToEndBox` is F(x, n) =
  ΔQ(s(x, n)) over the prefix attributes + natives (68 + 2 on BPIC2017,
  44 + 2 on BPIC2012), with `shapley_sampling` against the pool background
  and `deletion_test_e2e`; `channel_test` is the path patching. The effect
  arms' TreeSHAP is interventional against 100 pool prefixes; CatBoost only
  supports the path-dependent explainer, so the risk channel keeps it and the
  suite reports the gap.
- **Reward sign.** `Proba_if_Treated` / `Proba_if_Untreated` are
  P(undesired | arm). The reward's `y1` / `y0` are the *desired* outcome,
  `compose.desired_outcome(p) = 1[p < 0.5]`, and the positive-effect rule is
  `compose.positive_effect_rule` (p_T < 0.5 ≤ p_U). Shoush & Dumas's released
  code thresholds `p > 0.5` on the same columns; the shipped CSVs (and so the
  `released` / `cate` variants) keep that inverted reward. Never threshold
  these probabilities by hand: call the two helpers.
- **Results.** Scripts print a human-readable summary and write JSON;
  figures go to `figures/out/` unless `TIMING_PAPER_FIGURES` /
  `TIMING_XAI_FIGURES` redirect them. Numbers quoted in the paper come from
  the committed JSONs; if a change alters a number, regenerate the JSONs and
  `numbers.tex` in the same change.

## Layout

- `reproduce.sh` — the one-command reproduction (see [Quick start](#quick-start)).
- `paths.py` — every input and output path, overridable by environment
  variable (`TIMING_BPIC2012_CSV`, `TIMING_BPIC2017_CSV`,
  `TIMING_SIMBANK_RAW`, `TIMING_SIMBANK_PKL`, `TIMING_PAPER_FIGURES`).
- `pools.py` — the state pools every script evaluates on, defined once:
  the evaluation pool, one random decision point per test case (the paper's
  pool, Section 4.2: 3217 decisions to act and 4636 to wait on BPIC2017, 662
  and 510 on BPIC2012), and the full decision-point pools of the gain table.
  Its docstring states the two
  `available_resources` conventions and why SimBank is evaluated at every
  event.
- `dual_level.py`, `test_dual_level.py` — the differentiable heads
  (`MarginHead`, `WaitMarginHead`, ...), Integrated Gradients, and the
  `deletion_test` fidelity protocol.
- `foreign/train_ppo_fast_rl_prescriptive_monitoring.py` — the
  environment (Shoush & Dumas's state and reward, reimplemented from their
  CSV) and training script that produced the BPIC2012 and BPIC2017
  checkpoints.
- `models/` — the retrained BPIC2012 and BPIC2017 checkpoints, each with
  a `_manifest.json` (seed, step count, provenance), `_monitor.csv`, and
  `_training_curve.csv`.
- `simbank_resources/` — the SimBank Time-contact-HQ checkpoint and its
  own resource-augmentation script (`build_resources.py`, a synthetic
  capacity extension not part of SimBank's original design; see its
  docstring), training script (`train_ppo_simbank.py`), and a small
  reward-parameter diagnostic (`tune_cost.py`).
- `fidelity_test.py` — the deletion test (guided / random / anti-guided
  masking) on the timing explanation, wait and act sides, across the
  checkpoints, plus the per-feature share of mean |φ| (the paper's Table 4);
  writes `fidelity_results.json`.
- `compute_gain_table.py` — mean reward per decision point under the
  historically recorded action versus the retrained policy, across all
  three logs on the test cases, plus the validation rows the agents were
  chosen on (`bpic2012_val`, `bpic2017_val`); writes `gain_table_results.json`.
- `figures/revision-icpm2027/` — the figures and per-decision tests of the
  revised paper's running example (A and W), computed from the stored models
  without retraining; see its own README.
- `paper_numbers.py` — writes every number quoted in the paper as LaTeX
  macros (`numbers.tex`) from the committed result JSONs;
  `running_example_numbers.py` does the same for the running example.
- `figures/make_figures.py` — figures of the first submission (MDP timeline
  diagram, training-curve plot, two explanation cards); superseded in the
  revised paper by `figures/revision-icpm2027/`.
- `run_xai_suite.py`, `xai_methods.py`, `xai_metrics.py`, `xai_plots.py`,
  `test_xai_suite.py` — the explanation methods and explanation-quality
  metrics of the PPM-explainability literature, applied to the same
  checkpoints and the same evaluation pool (see below); writes
  `xai_suite_results.json` and `figures/out/xai/<log>/`.
- `risk_model.py`, `run_risk_suite.py`, `test_risk_model.py` — the risk
  level: the CatBoost outcome predictor behind the policy's
  `reliability`/`deviation` features, retrained per log with Shoush &
  Dumas's own split, encoding and settings (`foreign/common_files/` is
  their code), explained with TreeSHAP and evaluated with the same
  literature metrics plus the deletion test; two-level cards (risk +
  timing) for chosen decision points. Writes `models/risk/`,
  `risk_results.json`, `figures/out/risk/<log>/`.
- `effect_model.py`, `run_effect_suite.py` — the effect level: the
  CausalLift two-model estimator (XGBoost per treatment arm with IPW from
  a logistic propensity) behind the policy's `Proba_if_Treated` /
  `Proba_if_Untreated` features, retrained per BPIC log with Shoush &
  Dumas's split and encoding, explained with TreeSHAP per arm and
  attribute-then-subtract for the effect, evaluated with the deletion test;
  three-level cards (risk | effect | timing). Writes `models/effect/`,
  `effect_results.json`, `figures/out/effect/<log>/`. On SimBank the
  estimator is fit on the four events both arms share (the log's recording
  policy contacts HQ always at event 5 and skips at event 9/10) and the
  untreated arm is constant (skipping always cancels);
  `simbank_resources/add_effect_features.py` scores every event into the
  effect-augmented pkl the SimBank `cate` policy is trained and evaluated on.
- `compose.py`, `run_compose_suite.py`, `test_compose.py` — the composition
  study (paper Sections 3.4–3.5 and 5.2–5.3, Tables 5–6): the state rebuilt
  from the retrained models, the propagation of `φ^ΔQ` to the case attributes
  through `φ^r`, `φ^{p_T}`, `φ^{p_U}` with the path (channel) of every
  contribution (`propagate`; `FALLBACK_RATIO = 0.05` is the ε of the paper's
  Assumption 1), path patching (`channel_test`), the end-to-end function
  F(x, n) = ΔQ(s(x, n)) with a Shapley-sampling direct attribution and the
  deletion test on it, the (risky × treatable × acts) typology and the
  risk-vs-effect agreement on the shared prefix vocabulary. Writes
  `compose_results.json`, `figures/out/compose/<log>/`.
  The direct attribution of F is Shapley sampling against the pool as
  background (the same reference the levels use); `compose.anchor_to_direct`
  is the per-decision safeguard (direct ranking, propagated channel split),
  `compose.cancellation` the per-attribute cancellation ratio (Kramár et al. 2024),
  `compose.well_defined` / `baseline_alignment` the two conditions of the
  operator, `sign_groups` the agree / oppose / independent split of the
  risk-vs-effect signs. `--variant risk_retrained` runs it on the four-feature
  policy (same recipe, no effect features; stale, predates the reward-sign fix
  and is not in the paper), the composition with a live risk channel
  (`compose_results_risk_retrained.json`).
- `build_retrained_state.py` — the coherent-pipeline RL CSVs
  per split: the training split scored out of fold by `crossfit.py`
  (`data/retrained_state_<log>_train.csv`, the only state the agents train
  on), the validation split (`_val.csv`, where `select_policy.py` chooses
  each agent) and the test split (`.csv`, used once at the end for the gain
  and every explanation). `leakage_audit.py` checks the protocol.
- `figures/make_composition_flow.py` — the first submission's flow figure:
  the propagated card of one decision, case attributes → state features by
  explanation → margin (the revised paper's Fig. 4 is
  `figures/revision-icpm2027/scripts/make_fig6.py`).
- `models/variants/` — policy checkpoints that depart from the released
  design (`pools.VARIANTS`). Both append the causal model's counterfactual
  outcome probabilities to the state; `cate` trains with one case per
  episode (`--episode case`), `cate_stream` with the released stream
  episodes (kept as the negative result: it never intervenes either, see
  below). `fidelity_test.py --variant <v>` and `compute_gain_table.py
  --variant <v>` evaluate them.

## All commands

The full list of entry points (the [Quick start](#quick-start) is the
short path). Activate the environment first; retraining steps rewrite
committed models.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest test_dual_level.py -v
python fidelity_test.py            # deletion test + feature shares, all three logs
python compute_gain_table.py       # historical-vs-policy gain recomputation
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_figures.py
pytest test_xai_suite.py test_risk_model.py -v
python run_xai_suite.py            # literature XAI methods + metrics, all three logs (~10 min)
python risk_model.py               # retrain the outcome (risk) predictors -> models/risk/ (BPIC2017 takes ~1 h)
python run_risk_suite.py           # risk explanations + metrics + two-level cards -> risk_results.json
python effect_model.py             # retrain the causal-effect estimators -> models/effect/ (BPIC2017 takes ~75 min)
python run_effect_suite.py         # effect explanations + deletion test + three-level cards -> effect_results.json
python build_retrained_state.py    # RL CSVs per split (train out of fold, val, test) -> data/retrained_state_<log>[_train|_val].csv (needs the prepared logs)
python select_policy.py            # choose each agent on validation among models/variants/sweep_oof/ -> policy_selection.json
python run_compose_suite.py        # composition study -> compose_results.json, figures/out/compose/
TIMING_PAPER_FIGURES=<paper>/figures python paper_numbers.py   # every number in the paper -> <paper>/numbers.tex
# revised paper's running example (A and W) and Figs. 3-4: see figures/revision-icpm2027/README.md
python figures/revision-icpm2027/scripts/build_cards.py        # -> datos/cards_running_example.json
python figures/revision-icpm2027/scripts/make_fig5.py          # Fig. 3 (A and W explained alone)
python figures/revision-icpm2027/scripts/make_fig6.py          # Fig. 4 (composed explanations of A and W)
# first-submission figures (superseded):
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_three_level.py
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_composition_flow.py
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_mockup.py             # web mock-up (never in the paper)
```

The figure scripts read `risk_results.json` and `effect_results.json` and take their
plain-language feature readings from `figures/plain.py`; `run_effect_suite.py` also
writes a technical three-level card per log under `figures/out/effect/<log>/`.

Risk model (retrained, `models/risk/*_manifest.json`): test AUC 0.69 on BPIC2017
(early stop at iteration 117 of 1,000, 68 case attributes) and 0.63 on BPIC2012.
Effect model (retrained, `models/effect/*_manifest.json`): per-arm test AUC, each on
its own arm (treated / untreated), 0.62 / 0.72 on BPIC2017 and 0.58 / 0.66 on
BPIC2012; 20 % of BPIC2017's training prefixes are treated. The retrained p_T, p_U
correlate 0.26 / 0.38 with the shipped columns on BPIC2012 and not at all on BPIC2017
(the shipped file comes from a different run). Deletion test on the effect log-odds
ratio (`effect_results.json`): gap 51 / 38 SE (BPIC2012, k = 1 / 3) and 106 / 85 SE
(BPIC2017), anti-guided 0; top attributes FirstWithdrawalAmount_max 8 %, Activity 7 %,
RequestedAmount 6 % on BPIC2017; open_cases_max 29 %, open_cases_min 9 %, AMOUNT_REQ 8 %
on BPIC2012. SimBank (not in the paper; fit on events 1-4 of the 93,337 cases that
reach the contact/skip decision, 186,629 training prefixes, 37 % treated): treated-arm
AUC 0.95, untreated arm constant (p_U = 1: every skipped case is cancelled); deletion
gap 123 / 137 SE (k = 1 / 3), the good-outcome verdict 1[p_T < 0.5] flipped on 41 % /
71 % of the prefixes; top attributes amount 66 %, est_quality_max 17 %. Timing fidelity
on the SimBank `cate` policy (`fidelity_results.json`): act side 7,470 pool states,
wait side 2,530 (deleting the top feature reverses every wait decision); p_T carries 58 % / 81 % of the
attribution, reliability 22 % / 12 %.

## Findings about the released data that the code depends on

- The RL CSVs' `reliability` and `deviation` are outputs of the released
  offline phase's CatBoost outcome predictor (ensemble size 1):
  `deviation = 1 - predicted`, with class 1 = the *undesired* outcome
  (application not accepted), so `deviation = 1` means "predicted to end
  well". On BPIC2012 `reliability` is identical to `deviation` (ensemble
  agreement of one model); on BPIC2017 it is the probability of the
  predicted class, and `deviation = 1` on 99.2 % of rows.
- The released predictive-model configuration feeds CatBoost the column
  `time_to_event_m`, the time remaining until the case's treatment event
  (future information). `risk_model.py` drops it (`LEAK_NOTE`): with it the
  BPIC2017 predictor reaches AUC 0.94 with 71 % of its importance on that
  one column.
- The shipped BPIC2017 RL CSV does not come from the same predictor run as
  the prepared log: only 586k of its 1.2M rows match a (case, prefix) of the
  prepared log and 22 % of those disagree on the label, so its
  `predicted_proba_1` cannot be reproduced; BPIC2012's CSV is exactly the
  prepared log's test split (labels match on all 28,692 rows).
- Under the released 4-feature state and reward, always-wait is the
  optimal policy: intervening pays only when `y1 - y0 > 0` (19 % of
  BPIC2012 rows, 3 % of BPIC2017 rows), the reward's break-even is at
  P(positive effect) ≈ 0.6, and no observable cell of the state exceeds
  0.27. Whether `y1 - y0 > 0` is a deterministic function of
  `Proba_if_Treated`/`Proba_if_Untreated`, which the released state omits;
  the `cate` variants add them.
- Adding that information is not enough under the released *episode*
  structure: an episode streams the timestamp-sorted events of all cases and
  ends when any case finishes or the agent intervenes, and not intervening
  pays +50/+100 on every row whose effect is <= 0, so intervening forfeits
  the positive rewards of unrelated cases and `cate_stream` never
  intervenes (P(intervene) = 0.0000 on every row). `--episode case` makes
  the episode one case, the MDP the paper states; a per-case oracle then
  beats always-wait in 35 % of BPIC2012 cases (mean return 927 vs 594) and
  3.6 % of BPIC2017 cases (2086 vs 1969).
- Even then SB3's default `ent_coef=0` collapses the actor to
  never-intervene within the first few thousand steps (`cate_noent`):
  random interventions are punished on the 81-97 % of rows whose effect is
  <= 0 before the state can tell those rows apart, and the actor never
  explores again, although its critic learns that positive-effect states
  are worth less (V = 28 vs 227 on BPIC2012). `ent_coef=0.02` on the
  unscaled ±100 rewards collapses the same way (`cate_ent002`). What works
  (`cate`) is `--reward-scale 0.01 --ent-coef 0.3`: entropy stays near 0.2
  nats and the deterministic policy intervenes on 20.1 % of BPIC2012's
  decision points (precision 0.61, recall 0.63 against the positive-effect
  rows; gain 35.6 vs 24.3 always-wait, oracle 58.1) and 2.5 % of
  BPIC2017's (precision 0.98, recall 0.79; 55.9 vs 51.7, oracle 57.1).
  `fidelity_results_cate.json` holds the wait- and intervene-side deletion
  tests; the treatment-effect features carry 55-63 % of |φ| on BPIC2012.
  Evaluation always uses the unscaled reward and state.

## Literature explanation methods and metrics (`run_xai_suite.py`)

The suite reproduces, on this policy's timing decision, what five
PPM-explainability code bases produce for outcome predictors:

| Reference | Methods reproduced | Metrics reproduced |
|---|---|---|
| Warmuth & Leopold 2022 (text-based XPPM) | SHAP bar / beeswarm / waterfall / force | parsimony, monotonicity (Spearman), rediscovery rate (identity matching) |
| Elkhawaga et al. 2022 (PPM_XAI_Comparison) | SHAP global + local (force, decision, dependence), LIME + CSI/VSI stability, permutation importance (10 repeats, box plots), ALE, correlation heat map, mutual information, KMeans-selected local instances, timings | cross-method agreement (Spearman / Kendall / top-k Jaccard) |
| Stevens et al. (Quantifying-Explainability) | SHAP summary | parsimony, functional complexity (flip rate), monotonicity (Spearman and Kendall) |
| Elkhawaga et al. 2023 (ConsisXAI) | SHAP, permutation | reducts / core from nine scoring criteria, reduct- and core-consistency ratio (score-weighted and count-based), AIC / BIC |
| Stevens & De Smedt (X-MOP guidelines) | SHAP, permutation effects with observed-value replacement | parsimony per attribute family, functional complexity (L1) per family, monotonicity, LOD |

This suite is not reported in the revised paper. Every method explains the
same target, by default the wait-side margin `φ^ΔQ_wait` on the wait states
of the evaluation pool (`--target`
switches to the intervene margin, `p_wait`, `p_intervene` or the critic).
The paper's Integrated Gradients attribution is scored with the same
metrics, and every method's ranking is put through the paper's deletion
test (`k=1`), so both evaluation traditions appear side by side in
`xai_suite_results.json`. The state's three attribute families
(`progress` = relative_position, `prediction` = reliability and deviation,
`capacity` = available_resources) stand in for the event / case /
control-flow families of the reference metrics. Two deliberate deviations
from the reference code are documented in the docstrings: LIME uses the
continuous surrogate by default (`--lime-discretize` restores LIME's
default, which fits some states here with R² below 0.01), and the
weight-of-evidence criterion uses the usual +0.5 count adjustment so a
perfectly separating feature is not dropped.

## Data

The scripts read inputs that are too large for this repository:

| Input | Default location | Source | Override |
|---|---|---|---|
| `retrained_state_<log>_train.csv`, `_val.csv`, `.csv` (BPIC2012, BPIC2017, Sepsis) | `data/` | Zenodo archive (`retrained_states.tar.gz`, with SHA-256 sums), or regenerate with `build_retrained_state.py` from the prepared logs (~40 min) | `TIMING_<LOG>_RETRAINED[_TRAIN\|_VAL]_CSV`, `TIMING_SEPSIS_STATE[_TRAIN\|_VAL]_CSV` |
| `ready_to_use_adaptive_bpic2012.csv` | `data/` | Zenodo archive (see the paper's Data Availability) | `TIMING_BPIC2012_CSV` |
| `ready_to_use_adaptive_bpic2017.csv` | `data/` | Shoush & Dumas's repository, `rl/data/` (https://github.com/mshoush/RL-prescriptive-monitoring) | `TIMING_BPIC2017_CSV` |
| `prepared_treatment_outcome_time_to_event_<log>.csv` | `PeerJ2023/prepared_data/<log>/` | Shoush & Dumas's prepared logs (Zenodo archive, `PeerJ2023.zip`) | `TIMING_PREPARED_LOGS` |
| `sepsis_events.parquet` | `data/` | "Sepsis Cases - Event Log" (Mannhardt, 4TU.ResearchData, doi:10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460), one row per event with columns `case_id`, `activity`, `timestamp`, `org:group` and the event attributes | `TIMING_SEPSIS_PARQUET` |
| `simbank_time_contact_hq_with_resources.pkl` | `simbank_resources/data/` | Zenodo archive, or `simbank_resources/build_resources.py --n-servers 5` from SimBank's as-generated `loan_log_[_time_contact_HQ_]_100000_train_normal` (in `data/` or `TIMING_SIMBANK_RAW`) | `TIMING_SIMBANK_PKL` |
| `simbank_time_contact_hq_with_effect.pkl` | `simbank_resources/data/` | `simbank_resources/add_effect_features.py` (from the pkl above and `models/effect/SimBank_*`) | `TIMING_SIMBANK_EFFECT_PKL` |

## Retraining the checkpoints

Not needed to reproduce the results: the checkpoints are committed. The paper's
timing models are the sweep below, then `select_policy.py`:

```bash
# per log, e in 0.1 0.3 0.5 (plus 0.7 1.0 on the BPIC logs)
python foreign/train_ppo_fast_rl_prescriptive_monitoring.py --csv data/retrained_state_bpic2017_train.csv \
    --out models/variants/sweep_oof/ppo_bpic2017_ent$e --timesteps 600000 --save-at 300000 --episode case \
    --random-resources --extra-features Proba_if_Treated Proba_if_Untreated --ent-coef $e --reward-scale 0.01 --threads 1
python select_policy.py
```

Released four-feature design and SimBank:

```bash
python foreign/train_ppo_fast_rl_prescriptive_monitoring.py                     # BPIC2017, 300k steps, seed 42
python foreign/train_ppo_fast_rl_prescriptive_monitoring.py \
    --csv data/ready_to_use_adaptive_bpic2012.csv \
    --out models/ppo_bpic2012_rl_prescriptive_monitoring                         # BPIC2012
python simbank_resources/train_ppo_simbank.py                                    # SimBank, 150k steps, seed 42
```

For every log, the full sweep (the times are those recorded in the
manifests: about 50–75 min per 600k-step run, one CPU thread each; the runs are
independent and can run in parallel):

```bash
for log in bpic2012 bpic2017; do for e in 0.1 0.3 0.5 0.7 1.0; do
  python foreign/train_ppo_fast_rl_prescriptive_monitoring.py --csv data/retrained_state_${log}_train.csv \
      --out models/variants/sweep_oof/ppo_${log}_ent$e --timesteps 600000 --save-at 300000 --episode case \
      --random-resources --extra-features Proba_if_Treated Proba_if_Untreated --ent-coef $e --reward-scale 0.01 --threads 1
done; done
for e in 0.1 0.3 0.5; do   # Sepsis: always pass --csv (the default is the test state)
  python sepsis_resources/train_ppo_sepsis.py --csv data/retrained_state_sepsis_train.csv \
      --out models/variants/sweep_oof/ppo_sepsis_ent$e --timesteps 600000 --save-at 300000 --ent-coef $e --threads 1
done
python select_policy.py
```

Stopping `effect_model.py` with an unanchored `pkill -f effect_model` also
kills the calling shell; use `pkill -f '^python effect_model\.py'` (or the
`.venv/bin/python` prefix you launched it with).

## License

This repository (code, trained checkpoints, manifests, and result JSONs)
is released under CC0 1.0 (public domain dedication); see `LICENSE`.
