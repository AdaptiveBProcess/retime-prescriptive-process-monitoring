"""Paso (iv) sobre el par A/W: test de borrado y path patching de una sola decisión,
con las funciones del repo (compose.deletion_test_e2e y la lógica de compose.channel_test),
los modelos guardados y el mismo montaje que build_cards.py. No reentrena nada ni toca resultados del pool."""
import json, sys, os
from pathlib import Path
OUT = Path(sys.argv[1]).resolve()  # antes del chdir, para que una ruta relativa siga apuntando donde se pidió
REPO = Path(__file__).resolve().parents[3]  # raíz del repo (tests_pair/ -> revision-icpm2027/ -> figures/ -> repo)
sys.path.insert(0, str(REPO)); os.chdir(REPO)
import numpy as np
from stable_baselines3 import PPO
import compose as cp, effect_model as em, paths, pools, risk_model as rm
from run_compose_suite import _risk_shap
from run_effect_suite import EffectBox
from run_risk_suite import RiskBox
CASES = {"A": ("Application_381087716", 3), "W": ("Application_519115732", 3)}
LOG, VARIANT = "BPIC2017", pools.DEFAULT_VARIANT
states, rows, sfeats = pools.evaluation_pool(LOG, VARIANT)
clf, rfeats = rm.load_model(LOG); rbox = RiskBox(clf, rfeats)
arms, efeats = em.load_model(LOG); ebox = EffectBox(arms, efeats)
ppo = PPO.load(str(paths.variant_model(LOG, VARIANT)), device="cpu")
box = cp.EndToEndBox(LOG, rbox, ebox, ppo.policy, feats=sfeats)
names = box.columns
X, meta = rm.prefixes_for_pool(LOG, rows)
states_m = states[meta["pool_row"].to_numpy()]
r = rbox.proba(X); pT, pU = ebox.probs(X)
phi_r = _risk_shap(clf, X, rbox.columns)
bg = X.sample(n=min(100, len(X)), random_state=pools.POOL_SEED)
phiT, phiU, _ = ebox.shap_raw(X, background=bg)
lower = {"reliability": phi_r, "deviation": phi_r, "Proba_if_Treated": phiT, "Proba_if_Untreated": phiU}
S = cp.rebuild_state(LOG, states_m, r, pT, pU, sfeats); ref = S.mean(axis=0)
phi_t, sign = cp.timing_attribution(ppo.policy, S, ref)
m = cp.margin_of(ppo.policy, S)
prop = cp.propagate(phi_t, lower, sfeats)
Z = box.inputs(X, S); zref = box.reference(Z)
cid = meta["case_id"].astype(str).to_numpy(); pnr = meta["prefix_nr"].to_numpy()
D = len(box.raw_columns)
out = {}
for tag, (case_id, k) in CASES.items():
    i = int(np.flatnonzero((cid == case_id) & (pnr == k))[0])
    Zi = Z.iloc[[i]].reset_index(drop=True); si = sign[[i]]
    f0 = float(box.f(Zi, si)[0])
    dele = {}
    for kk in (1, 3):
        res = cp.deletion_test_e2e(box, Zi, zref, si, {"composed": prop["composed"][[i]]}, kk, seed=123)
        order = np.argsort(-np.abs(prop["composed"][i]))[:kk]
        fg = float(box.f(box.mask(Zi, order[None, :], zref), si)[0])
        dele[str(kk)] = {"masked": [names[j] for j in order], "f_after_guided": fg, "abs_guided": res["composed"]["abs_guided"],
                         "abs_anti": res["composed"]["abs_anti"], "abs_random": res["abs_random"], "flip_guided": res["composed"]["flip_guided"]}
    # path patching on the top composed prefix attribute (k=1), as channel_test does
    j = int(np.argsort(-np.abs(prop["composed"][i, :D]))[0]); col = box.raw_columns[j]
    Zm = Zi.copy(); Zm.iloc[0, Zm.columns.get_loc(col)] = zref[col]
    r0, pT0, pU0 = box.lower(Zi); rM, pTM, pUM = box.lower(Zm)
    rel = Zi["relative_position"].to_numpy(float); resv = Zi["available_resources"].to_numpy(float)
    sg = float(sign[i])
    mg = lambda a, b, c: float(sg * cp.margin_of(box.policy, cp.build_state(box.log, rel, a, resv, b, c, box.feats))[0])
    m0 = mg(r0, pT0, pU0)
    measured = {"risk": m0 - mg(rM, pT0, pU0), "effect_T": m0 - mg(r0, pTM, pU0), "effect_U": m0 - mg(r0, pT0, pUM)}
    traced = {c: float(prop["channels"][c][i, j]) for c in measured}
    canc = lambda d: (1 - abs(sum(d.values())) / sum(abs(v) for v in d.values())) if sum(abs(v) for v in d.values()) > 1e-12 else None
    out[tag] = {"case_id": case_id, "dq": float(m[i]), "f0_signed": f0, "deletion": dele,
                "patching": {"attribute": col, "case_value": (Zi[col].iloc[0] if isinstance(Zi[col].iloc[0], str) else float(Zi[col].iloc[0])), "ref_value": (zref[col] if isinstance(zref[col], str) else float(zref[col])),
                             "measured": measured, "traced": traced, "c_measured": canc(measured), "c_traced": canc(traced)},
                "fallback": {f: bool(cp._weights(lower[f])[1][i]) for f in ("reliability", "Proba_if_Treated", "Proba_if_Untreated")},
                "nondegeneracy_ratio": {f: float(abs(lower[f][i].sum()) / np.abs(lower[f][i]).sum()) for f in ("reliability", "Proba_if_Treated", "Proba_if_Untreated")}}
    print(tag, json.dumps(out[tag], indent=1, default=float), flush=True)
OUT.write_text(json.dumps(out, indent=2, default=float))
