"""Cards del running example: los dos casos del par mínimo
de BPIC2017, A (actúa) y W (espera), con el mismo esquema que las cards de
risk_results.json / effect_results.json / compose_results.json del repo.

Copia fuera del repo: importa el código del repo original sin modificarlo y repite
run_compose_suite.run_log (lectura "rebuilt") hasta la propagación sobre el pool
completo (la referencia y el fondo son los del pool). Añade:
  - la traza del prefijo (log preparado, mismo orden que figures/make_three_level.py),
  - la explicación directa de F con 200 permutaciones (el JSON del repo usa 20),
  - las cuotas por nivel de la explicación de timing (|phi| por nivel / suma),
  - el camino de riesgo del atributo de arriba, para el cálculo a mano.

Necesita el entorno completo del repo (torch, stable-baselines3, catboost, xgboost,
shap, captum). Uso, desde cualquier carpeta:

    python build_cards.py            # -> ../datos/cards_running_example.json (~5 min)
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # raíz del repo (scripts/ -> revision-icpm2027/ -> figures/ -> repo)
sys.path.insert(0, str(REPO))
import os  # noqa: E402

os.chdir(REPO)  # algunos módulos resuelven rutas relativas al repo

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import compose as cp  # noqa: E402
import effect_model as em  # noqa: E402
import paths  # noqa: E402
import pools  # noqa: E402
import risk_model as rm  # noqa: E402
import running_example_numbers as ren  # noqa: E402
from run_compose_suite import _flows, _risk_shap, _top  # noqa: E402
from run_effect_suite import EffectBox  # noqa: E402
from run_risk_suite import RiskBox  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "datos" / "cards_running_example.json"
CASES = {"A": ("Application_381087716", 3), "W": ("Application_519115732", 3)}
N_PERM, SEEDS, TOP = 200, (123, 7), 6
LOG, VARIANT = "BPIC2017", pools.DEFAULT_VARIANT


def main():
    states, rows, sfeats = pools.evaluation_pool(LOG, VARIANT)
    clf, rfeats = rm.load_model(LOG)
    rbox = RiskBox(clf, rfeats)
    arms, efeats = em.load_model(LOG)
    ebox = EffectBox(arms, efeats)
    ppo = PPO.load(str(paths.variant_model(LOG, VARIANT)), device="cpu")
    box = cp.EndToEndBox(LOG, rbox, ebox, ppo.policy, feats=sfeats)
    names = box.columns
    X, meta = rm.prefixes_for_pool(LOG, rows)
    keep = meta["pool_row"].to_numpy()
    states_m = states[keep]

    r = rbox.proba(X)
    pT, pU = ebox.probs(X)
    phi_r = _risk_shap(clf, X, rbox.columns)
    bg = X.sample(n=min(100, len(X)), random_state=pools.POOL_SEED)  # = run_compose_suite --n-background 100
    phiT, phiU, _ = ebox.shap_raw(X, background=bg)
    lower = {"reliability": phi_r, "deviation": phi_r, "Proba_if_Treated": phiT, "Proba_if_Untreated": phiU}
    S = cp.rebuild_state(LOG, states_m, r, pT, pU, sfeats)
    ref = S.mean(axis=0)
    phi_t, sign = cp.timing_attribution(ppo.policy, S, ref)
    m = cp.margin_of(ppo.policy, S)
    prop = cp.propagate(phi_t, lower, sfeats)
    fb = {f: cp._weights(lower[f])[1] for f in ("reliability", "Proba_if_Treated", "Proba_if_Untreated")}
    m_ref = float(cp.margin_of(ppo.policy, ref[None, :].astype(np.float32))[0])

    Z = box.inputs(X, S)
    zref = box.reference(Z)
    zbg = box.inputs(bg, states_m[bg.index.to_numpy()])
    risk_base = {x["log"]: x for x in json.loads(paths.RISK_JSON.read_text())["results"]}[LOG]["risk"]["expected_value_logodds"]

    cid = meta["case_id"].astype(str).to_numpy()
    pnr = meta["prefix_nr"].to_numpy()
    D = len(box.raw_columns)
    out = {"log": LOG, "variant": VARIANT, "reference_state": dict(zip(sfeats, ref.astype(float).tolist())),
           "margin_at_reference": m_ref, "n_perm_direct": N_PERM, "seeds_direct": list(SEEDS), "cards": {}}
    for tag, (case_id, k) in CASES.items():
        i = int(np.flatnonzero((cid == case_id) & (pnr == k))[0])
        xrow = X.iloc[i]
        val = lambda a: (xrow[a] if isinstance(xrow[a], str) else float(xrow[a]))  # noqa: E731
        cate = phiU[i] - phiT[i]
        o_r = np.argsort(-np.abs(phi_r[i]))[:TOP]
        o_e = np.argsort(-np.abs(cate))[:TOP]
        flows = _flows(i, names, prop, xrow, top=8)
        for f in flows:
            ch = [f["risk"], f["effect_T"], f["effect_U"]]
            tot = sum(abs(v) for v in ch)
            f["cancellation"] = 1 - abs(sum(ch)) / tot if tot else 0.0
        direct = {}
        for s in SEEDS:
            phi_d = cp.shapley_sampling(box, Z.iloc[[i]].reset_index(drop=True), zref, sign[[i]], n_perm=N_PERM, seed=s, background=zbg)[0]
            direct[str(s)] = {"top": _top(names, phi_d, TOP), "spearman_vs_composed": float(spearmanr(phi_d, prop["composed"][i])[0]),
                              "same_top": bool(np.argmax(np.abs(phi_d)) == np.argmax(np.abs(prop["composed"][i])))}
        a = np.abs(phi_t[i])
        lvl = {"risk": a[[sfeats.index("reliability"), sfeats.index("deviation")]].sum() / a.sum(),
               "effect": a[[sfeats.index("Proba_if_Treated"), sfeats.index("Proba_if_Untreated")]].sum() / a.sum(),
               "native": a[[sfeats.index("relative_position"), sfeats.index("available_resources")]].sum() / a.sum()}
        top_attr = flows[0]["input"]
        j = names.index(top_attr)
        mass = float(phi_t[i][sfeats.index("reliability")] + phi_t[i][sfeats.index("deviation")])
        hand = {"attribute": top_attr, "timing_risk_mass": mass, "phi_r_attr": float(phi_r[i][j]), "phi_r_sum": float(phi_r[i].sum()),
                "logit_r_minus_base": float(np.log(r[i] / (1 - r[i])) - risk_base),
                "path": mass * float(phi_r[i][j]) / float(phi_r[i].sum()), "stored_path": flows[0]["risk"]}
        prefix = ren._prefix(case_id, k)
        out["cards"][tag] = {
            "case_id": case_id, "prefix_nr": k, "pool_row": int(meta["pool_row"].iloc[i]),
            "action": "intervene" if m[i] > 0 else "wait", "dq": float(m[i]), "p_act": float(1 / (1 + np.exp(-m[i]))),
            "r": float(r[i]), "pT": float(pT[i]), "pU": float(pU[i]),
            "risky": bool(r[i] > 0.5), "positive_effect_rule": bool(cp.positive_effect_rule(np.array([pT[i]]), np.array([pU[i]]))[0]),
            "state": dict(zip(sfeats, S[i].astype(float).tolist())), "timing_phi": dict(zip(sfeats, phi_t[i].astype(float).tolist())),
            "timing_sum": float(phi_t[i].sum()), "level_shares": lvl,
            "fallback": {f: bool(v[i]) for f, v in fb.items()},
            "risk_top": [[names[j_], val(names[j_]), float(phi_r[i][j_])] for j_ in o_r],
            "effect_top": [[names[j_], val(names[j_]), float(cate[j_]), float(phiT[i][j_]), float(phiU[i][j_])] for j_ in o_e],
            "flows": flows, "composed_top": _top(names, prop["composed"][i], TOP),
            "native_phi": {"relative_position": float(prop["native"][i][0]), "available_resources": float(prop["native"][i][1])},
            "direct": direct, "hand_risk_path": hand,
            "trace": [{"activity": e["Activity"], "timestamp": e["time:timestamp"], "resource": e["org:resource"],
                       "open_cases": int(float(e["open_cases"]))} for e in prefix],
            "case_attributes": {a_: prefix[0][a_] for a_ in ("ApplicationType", "LoanGoal", "RequestedAmount")},
        }
        assert abs(hand["path"] - hand["stored_path"]) < 1e-6, (tag, hand)
        print(tag, case_id, "dq", round(float(m[i]), 3), "top", flows[0]["input"], round(flows[0]["total"], 3),
              "direct", {s: d["top"][0] for s, d in direct.items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print("cards ->", OUT)


if __name__ == "__main__":
    main()
