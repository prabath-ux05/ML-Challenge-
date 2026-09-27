"""Choose between runs and their average with out-of-fold F0.5 on train, then write the test submission.

    # 1) evaluate on train (can run before the runs have finished predicting on test)
    python src/blend_oof.py eval  --works work_v4 work_v5 --sample-seeds 7 11 --data-dir dataset_aug
    # 2) apply the chosen candidate to test
    python src/blend_oof.py apply --works work_v4 work_v5 --out-dir output_final [--candidate NAME]

Candidates: every single run and the average of all runs. For each, on train: union of the runs'
out-of-fold scores (a pair scored by only some runs gets the mean of the available scores) ->
out-of-fold singleton model -> decision tuned on held-out entities (in no run's fitting sample)
-> macro-F0.5 on those entities. Runs must share the fold assignment (fixed in pipeline.py).
Pairs are keyed by integer indices; runs whose training targets are ordered differently (e.g. one
trained without the synthetic decoys) are re-indexed through the entity ids.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (decide, load_truth, macro_f05, singleton_model, stratified_group_folds,  # noqa: E402
                      tune_decision, write_lists)


def _ids(work, f):
    return pl.read_parquet(os.path.join(work, "norm", f), columns=["id"])["id"]


def scores(work, split, ref):
    """(i, j, p[, y]) of one run, with j re-indexed to the reference run's target order if needed."""
    f, col = ("train_oof.parquet", "p2") if split == "train" else ("test_scores.parquet", "p")
    cols = ["i", "j", col] + (["y"] if split == "train" else [])
    d = pl.read_parquet(os.path.join(work, f), columns=cols).rename({col: "p"})
    mine, theirs = _ids(work, f"{split}_tg.parquet"), _ids(ref, f"{split}_tg.parquet")
    if len(mine) != len(theirs) or not (mine == theirs).all():
        m = (pl.DataFrame({"id": mine}).with_row_index("j")
             .join(pl.DataFrame({"id": theirs}).with_row_index("j2"), on="id").select("j", "j2"))
        d = d.join(m, on="j").drop("j").rename({"j2": "j"})
    return d.with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))


def union(frames, weights=None):
    weights = weights or [1.0] * len(frames)
    u = None
    for k, d in enumerate(frames):
        d = d.rename({"p": f"p{k}"})
        if u is None:
            u = d
        else:
            keys = ["i", "j"] + (["y"] if "y" in d.columns else [])
            u = u.join(d, on=keys, how="full", coalesce=True)
    num = pl.sum_horizontal([pl.col(f"p{k}").fill_null(0) * w for k, w in enumerate(weights)])
    den = pl.sum_horizontal([pl.col(f"p{k}").is_not_null().cast(pl.Float32) * w for k, w in enumerate(weights)])
    return u.with_columns(p=num / den).select("i", "j", "p", *(["y"] if "y" in u.columns else [])).sort("i", "j")


def candidates(works, spec=None):
    """name -> (works, weights). spec: list like ['work_v5', 'work_v4+work_v5', 'work+work_v4+2*work_v5']."""
    by = {os.path.basename(w.rstrip("/")): w for w in works}
    if not spec:
        spec = list(by) + (["+".join(by)] if len(works) > 1 else [])
    out = {}
    for s in spec:
        ws, wt = [], []
        for part in s.split("+"):
            k, _, n = part.rpartition("*")
            w, name = (float(k), n) if k else (1.0, part)
            ws.append(by[name])
            wt.append(w)
        out[s] = (ws, wt)
    return out


def cmd_eval(a):
    W, ref = a.works, a.works[-1]
    seeds = a.sample_seeds or [7] * len(W)
    s1n = pl.read_parquet(os.path.join(ref, "norm", "train_s1.parquet"), columns=["id", "country", "name_core"])
    truth = load_truth(a.data_dir, s1n.select("id"), pl.DataFrame({"id": _ids(ref, "train_tg.parquet")}))
    fold_s1 = stratified_group_folds(s1n, truth, a.folds)
    held = np.ones(s1n.height, bool)
    for sd in seeds:
        h = s1n.select(pl.col("id").hash(seed=sd)).to_series().to_numpy()
        held &= ~(((h // a.folds) % 1000) < int(a.sample * 1000))
    eval_i = pl.Series(np.where(held)[0].astype(np.uint32))
    if len(eval_i) > a.eval_s1:
        eval_i = eval_i.sample(a.eval_s1, seed=1)
    ev = pl.DataFrame({"i": eval_i})
    ev_truth = truth.join(ev, on="i", how="semi")
    country = s1n["country"].to_numpy()
    print(f"held-out entities: {len(eval_i)}", flush=True)
    report = {}
    tg_ref = _ids(ref, "train_tg.parquet")
    synth = pl.DataFrame({"j": np.where(tg_ref.str.contains("-D").to_numpy())[0].astype(np.uint32)})
    for name, (ws, wt) in candidates(W, a.spec).items():
        u = union([scores(w, "train", ref) for w in ws], wt)
        ij, p = u.select("i", "j"), u["p"].to_numpy()
        has_oof, sm = singleton_model(ij, p, u.filter(pl.col("y") == 1).select("i", "j"), s1n, fold_s1=fold_s1, K=a.folds)
        tj = ij.join(ev, on="i", how="semi").select("j").unique()
        keep = ij.with_row_index("r").join(tj, on="j", how="semi")["r"].to_numpy()
        sc, prm = tune_decision(ij[keep], p[keep], ev_truth, eval_i, seg=None, has_match=has_oof)
        pred = decide(ij[keep], p[keep], prm, None, has_oof)
        by_c = {c: round(macro_f05(pred, ev_truth, pl.Series(np.intersect1d(eval_i.to_numpy(), np.where(country == c)[0]).astype(np.uint32))), 4)
                for c in sorted(set(country))}
        # "original world": same entities and decision, synthetic decoys removed from the candidates
        ko = ij[keep].with_columns(p=pl.Series(p[keep])).join(synth, on="j", how="anti")
        f_orig = macro_f05(decide(ko.select("i", "j"), ko["p"].to_numpy(), prm, None, has_oof), ev_truth, eval_i)
        report[name] = dict(f05=sc, f05_orig=f_orig, params=prm, by_country=by_c, works=ws, weights=wt, sm=sm)
        print(f"{name}: held-out F0.5 test-like={sc:.4f}  original={f_orig:.4f}  {by_c}  {prm}", flush=True)
        del u, ij, p, has_oof
    best = max(report, key=lambda k: report[k]["f05"])
    print(f"BEST: {best} ({report[best]['f05']:.4f})")
    joblib.dump(dict(report=report, best=best), os.path.join(ref, "blend_eval.joblib"))


def cmd_apply(a):
    ref = a.works[-1]
    r = joblib.load(os.path.join(ref, "blend_eval.joblib"))
    name = a.candidate or r["best"]
    c = r["report"][name]
    u = union([scores(w, "test", ref) for w in c["works"]], c["weights"])
    ij, p = u.select("i", "j"), u["p"].to_numpy()
    s1t = pl.read_parquet(os.path.join(ref, "norm", "test_s1.parquet"), columns=["id", "name_core", "country"])
    hm = singleton_model(ij, p, None, s1t, models=c["sm"])[0]
    pred = decide(ij, p, c["params"], None, hm)
    assert pred.join(ij, on=["i", "j"], how="anti").height == 0
    os.makedirs(a.out_dir, exist_ok=True)
    tg = _ids(ref, "test_tg.parquet").to_list()
    write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1t["id"].to_list(), pred, tg, "matched_entity_ids")
    write_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1t["id"].to_list(), ij, tg, "candidate_entity_ids")
    st = (s1t.select("country").with_row_index("i").with_columns(pl.col("i").cast(pl.UInt32))
          .join(pred.group_by("i").len("n"), on="i", how="left").fill_null(0)
          .group_by("country").agg(links=pl.col("n").mean(), empty=(pl.col("n") == 0).mean()).sort("country"))
    print(f"{name}: wrote {a.out_dir} ({pred.height} links)  {st.to_dicts()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["eval", "apply"])
    ap.add_argument("--works", nargs="+", required=True, help="run folders; the last one is the reference")
    ap.add_argument("--data-dir", default="dataset_aug")
    ap.add_argument("--sample-seeds", nargs="+", type=int, default=None)
    ap.add_argument("--sample", type=float, default=0.3)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--eval-s1", type=int, default=300_000)
    ap.add_argument("--candidate", default=None, help="apply: candidate name (default: the best)")
    ap.add_argument("--spec", nargs="+", default=None, help="eval: candidates, e.g. work_v5 work_v4+2*work_v5")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    cmd_eval(a) if a.command == "eval" else cmd_apply(a)


if __name__ == "__main__":
    main()
