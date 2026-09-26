"""End-to-end entity-resolution pipeline.

    python src/pipeline.py train   --data-dir ../../dataset --model-dir models
    python src/pipeline.py predict --data-dir ../../dataset --model-dir models --out-dir ../../output
    python src/pipeline.py run     ... (train + predict)
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import Vectors, blocking_recall, generate_candidates, load_source, prepare  # noqa: E402
from features import context_features, pair_features  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_split(data_dir, split):
    d = os.path.join(data_dir, split)
    s1 = load_source(os.path.join(d, f"{split}_source1.tsv"))
    tg = pd.concat([load_source(os.path.join(d, f"{split}_source{k}.tsv")) for k in (2, 3)], ignore_index=True)
    return prepare(s1), prepare(tg)


def load_truth(data_dir):
    gt = pd.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    return {r.source1_entity_id.strip(): {x.strip() for x in r.matched_entity_ids.split(",") if x.strip()}
            for r in gt.itertuples()}


def build(s1, tg, block_cfg=None):
    log(f"records: S1={len(s1)} S2+S3={len(tg)}")
    vec = Vectors(s1, tg)
    cand = generate_candidates(s1, tg, vec, block_cfg)
    log(f"candidates: {len(cand)} pairs ({len(cand) / max(len(s1), 1):.1f} per S1)")
    F = pair_features(s1, tg, cand, vec)
    log(f"pair features: {F.shape}")
    cand = cand.assign(s1_id=s1.id.values[cand.i.values], tg_id=tg.id.values[cand.j.values])
    return vec, cand, F


# ---------------------------------------------------------------------------
# metric + decision
# ---------------------------------------------------------------------------
def f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(preds: dict, truth: dict, ids) -> float:
    return float(np.mean([f05(preds.get(s, set()), truth.get(s, set())) for s in ids]))


def decide(cand: pd.DataFrame, p: np.ndarray, s1_ids, params: dict) -> dict:
    d = cand[["s1_id", "tg_id"]].assign(p=p)
    if params.get("one_to_one", True):
        d = d[d.p >= d.groupby("tg_id")["p"].transform("max") - 1e-9]
    pool = d.groupby("s1_id")["p"].apply(lambda x: np.clip(x.values, 0, 1)).to_dict()
    d = d[d.p >= params["floor"]]
    out = {s: set() for s in s1_ids}
    if params["mode"] == "thr":
        for s, t in zip(d.s1_id.values, d.tg_id.values):
            out[s].add(t)
        return out
    # expected-F0.5 maximisation per S1 entity (candidates as independent Bernoullis)
    d = d.sort_values(["s1_id", "p"], ascending=[True, False])
    for s, grp in d.groupby("s1_id", sort=False):
        ps = np.clip(grp.p.values * params.get("scale", 1.0), 0, 1)
        tot = pool[s].sum()
        best_k, best = 0, float(np.prod(1 - pool[s]))  # predicting empty = P(no match at all)
        tp = 0.0
        for k in range(1, len(ps) + 1):
            tp += ps[k - 1]
            fp, fn = k - tp, max(tot - tp, 0)
            ef = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
            if ef > best:
                best, best_k = ef, k
        out[s] = set(grp.tg_id.values[:best_k])
    return out


def tune_decision(cand, p, truth, s1_ids):
    best = None
    for o2o in (True, False):
        for thr in np.round(np.arange(0.2, 0.96, 0.025), 3):
            prm = dict(mode="thr", floor=float(thr), one_to_one=o2o)
            sc = macro_f05(decide(cand, p, s1_ids, prm), truth, s1_ids)
            if best is None or sc > best[0]:
                best = (sc, prm)
        for floor in (0.02, 0.1, 0.2, 0.3, 0.4):
            prm = dict(mode="ef", floor=floor, one_to_one=o2o)
            sc = macro_f05(decide(cand, p, s1_ids, prm), truth, s1_ids)
            if sc > best[0]:
                best = (sc, prm)
    return best


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def make_model(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=800, max_leaf_nodes=48, min_samples_leaf=30,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=seed)


def cv_fit(X, y, groups, folds, seed=0):
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        m = make_model(seed + k).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
    return oof, models


def predict_avg(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_train(a):
    s1, tg = load_split(a.data_dir, "train")
    truth = load_truth(a.data_dir)
    s1_ids = s1.id.tolist()
    tg_owner = {}
    for s, ts in truth.items():
        for t in ts:
            tg_owner.setdefault(t, []).append(s)
    multi = sum(len(v) > 1 for v in tg_owner.values())
    log(f"truth: {sum(map(len, truth.values()))} links, singletons={sum(not v for v in truth.values())}"
        f"/{len(truth)}, targets linked to >1 S1: {multi}")

    vec, cand, F = build(s1, tg, a.block_cfg)
    y = np.array([t in truth.get(s, ()) for s, t in zip(cand.s1_id, cand.tg_id)], dtype=np.int8)
    cset = cand.groupby("s1_id")["tg_id"].apply(set).to_dict()
    log(f"blocking recall ceiling: {blocking_recall(cset, truth):.4f}  positives in cands={y.sum()}")
    ceiling = macro_f05({s: cset.get(s, set()) & truth.get(s, set()) for s in s1_ids}, truth, s1_ids)
    log(f"oracle macro-F0.5 given candidates: {ceiling:.4f}")

    groups = cand.s1_id.values
    oof1, m1 = cv_fit(F, y, groups, a.folds)
    log("stage-1 OOF done")
    C = context_features(cand, oof1, vec)
    X2 = pd.concat([F, C], axis=1)
    oof2, m2 = cv_fit(X2, y, groups, a.folds, seed=100)
    log("stage-2 OOF done")

    res = {}
    for name, p in (("stage1", oof1), ("stage2", oof2)):
        sc, prm = tune_decision(cand, p, truth, s1_ids)
        res[name] = (sc, prm)
        log(f"{name}: OOF macro-F0.5={sc:.4f} with {prm}")
    stage = "stage2" if res["stage2"][0] >= res["stage1"][0] else "stage1"
    params = res[stage][1]
    preds = decide(cand, oof2 if stage == "stage2" else oof1, s1_ids, params)
    for c in sorted(s1.country.unique()):
        ids = s1.id[s1.country == c].tolist()
        log(f"  country={c!r}: n={len(ids)} F0.5={macro_f05(preds, truth, ids):.4f}")
    sing = [s for s in s1_ids if not truth.get(s)]
    nons = [s for s in s1_ids if truth.get(s)]
    log(f"  singletons F0.5={macro_f05(preds, truth, sing):.4f}  matched-entities F0.5={macro_f05(preds, truth, nons):.4f}")

    os.makedirs(a.model_dir, exist_ok=True)
    joblib.dump(dict(m1=m1, m2=m2, stage=stage, params=params, cols1=list(F.columns), cols2=list(X2.columns),
                     block_cfg=a.block_cfg), os.path.join(a.model_dir, "model.joblib"))
    with open(os.path.join(a.model_dir, "cv_report.json"), "w") as f:
        json.dump({k: {"f05": v[0], "params": v[1]} for k, v in res.items()} | {"chosen": stage}, f, indent=2)
    if a.dump_oof:
        cand.assign(y=y, p1=oof1, p2=oof2).to_csv(os.path.join(a.model_dir, "oof_pairs.tsv"), sep="\t", index=False)
    log(f"saved model to {a.model_dir}")


def write_list_tsv(path, s1_ids, mapping, col):
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            f.write(f"{s}\t{','.join(sorted(mapping.get(s, ())))}\n")


def cmd_predict(a):
    bundle = joblib.load(os.path.join(a.model_dir, "model.joblib"))
    s1, tg = load_split(a.data_dir, "test")
    vec, cand, F = build(s1, tg, bundle["block_cfg"])
    F = F[bundle["cols1"]]
    p1 = predict_avg(bundle["m1"], F)
    p = p1
    if bundle["stage"] == "stage2":
        X2 = pd.concat([F, context_features(cand, p1, vec)], axis=1)[bundle["cols2"]]
        p = predict_avg(bundle["m2"], X2)
    s1_ids = s1.id.tolist()
    preds = decide(cand, p, s1_ids, bundle["params"])
    cset = cand.groupby("s1_id")["tg_id"].apply(set).to_dict()
    os.makedirs(a.out_dir, exist_ok=True)
    write_list_tsv(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, preds, "matched_entity_ids")
    write_list_tsv(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1_ids, cset, "candidate_entity_ids")
    if a.dump_scores:
        cand.assign(p=p).to_csv(os.path.join(a.out_dir, "test_pair_scores.tsv"), sep="\t", index=False)
    n_match = sum(bool(v) for v in preds.values())
    log(f"wrote outputs to {a.out_dir}: {n_match}/{len(s1_ids)} S1 with matches, "
        f"{sum(map(len, preds.values()))} links")
    for c in sorted(s1.country.unique()):
        ids = s1.id[s1.country == c]
        log(f"  country={c!r}: n={len(ids)} matched-rate={np.mean([bool(preds[s]) for s in ids]):.3f}")
    self_check(s1_ids, set(tg.id), preds, cset)


def self_check(s1_ids, tg_ids, preds, cset):
    assert len(set(s1_ids)) == len(s1_ids), "duplicate S1 ids"
    for s, ts in preds.items():
        assert ts <= tg_ids, f"{s}: unknown target ids"
        assert ts <= cset.get(s, set()), f"{s}: match not in candidates"
        assert all(t[:3] in ("S2-", "S3-") for t in ts)
    log("self-check PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict", "run"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-cfg", type=json.loads, default=None, help='JSON, e.g. {"k_name":40}')
    ap.add_argument("--dump-oof", action="store_true")
    ap.add_argument("--dump-scores", action="store_true")
    a = ap.parse_args()
    if a.command in ("train", "run"):
        cmd_train(a)
    if a.command in ("predict", "run"):
        cmd_predict(a)


if __name__ == "__main__":
    main()
