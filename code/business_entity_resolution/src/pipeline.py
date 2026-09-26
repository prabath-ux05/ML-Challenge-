"""End-to-end entity-resolution pipeline (scales to the full 2M x 10M challenge data).

    python src/pipeline.py train   --data-dir dataset --work-dir work
    python src/pipeline.py predict --data-dir dataset --work-dir work --out-dir output
    python src/pipeline.py run     ...   (train + predict)

Stages: normalise (parallel, cached) -> key blocking -> pair features (chunked parquet)
-> group features -> stage-1 GBDT (K folds, OOF on every train pair) -> context features
-> stage-2 GBDT -> F0.5-optimal decision (tuned on OOF) -> TSV outputs.
"""
import argparse
import glob
import json
import os
import sys
import time

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import DEFAULT_BLOCKING, blocking_recall, generate_candidates  # noqa: E402
from features import (GROUP_BASE, context_features, group_features, name_frequencies,  # noqa: E402
                      pair_features)
from prep import load_split  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# cached stages
# ---------------------------------------------------------------------------
def candidates(split, s1, tg, work, block_cfg):
    path = os.path.join(work, f"{split}_cand.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    cand = generate_candidates(s1, tg, block_cfg).sort("i", "j")
    cand.write_parquet(path)
    return cand


def feature_parts(split, s1, tg, cand, work, chunk=3_000_000):
    """Pair features written as row-aligned parquet parts + one group-feature file."""
    fdir = os.path.join(work, f"{split}_feat")
    done = os.path.join(fdir, "_DONE")
    if not os.path.exists(done):
        os.makedirs(fdir, exist_ok=True)
        f1, ft = name_frequencies(s1, tg)
        slim = []
        for k, s in enumerate(range(0, cand.height, chunk)):
            F = pair_features(s1, tg, cand.slice(s, chunk), f1, ft)
            F.write_parquet(os.path.join(fdir, f"part_{k:04d}.parquet"))
            slim.append(F.select([c for c in GROUP_BASE if c in F.columns]))
            log(f"  features {split}: {min(s + chunk, cand.height)}/{cand.height}")
        slim = pl.concat([cand.select("i", "j"), pl.concat(slim)], how="horizontal")
        group_features(slim).write_parquet(os.path.join(fdir, "group.parquet"))
        open(done, "w").close()
    return sorted(glob.glob(os.path.join(fdir, "part_*.parquet"))), os.path.join(fdir, "group.parquet")


def iter_X(parts, group_path, extra=None, rows=None):
    """Yield (offset, X) chunks = pair features + group features (+ extra columns)."""
    G = pl.read_parquet(group_path)
    off = 0
    for p in parts:
        F = pl.read_parquet(p)
        n = F.height
        X = pl.concat([F, G.slice(off, n)] + ([extra.slice(off, n)] if extra is not None else []), how="horizontal")
        if rows is not None:
            X = X.filter(rows[off:off + n])
        yield off, n, X
        off += n


# ---------------------------------------------------------------------------
# metric + decision (vectorised)
# ---------------------------------------------------------------------------
def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_idx: pl.Series) -> float:
    """pred/truth: (i, j) pairs. s1_idx: all S1 indices being evaluated (singletons included)."""
    base = pl.DataFrame({"i": s1_idx.cast(pl.UInt32)})
    tp = pred.join(truth, on=["i", "j"], how="semi").group_by("i").agg(pl.len().alias("tp"))
    npred = pred.group_by("i").agg(pl.len().alias("np"))
    ntrue = truth.group_by("i").agg(pl.len().alias("nt"))
    d = (base.join(tp, on="i", how="left").join(npred, on="i", how="left").join(ntrue, on="i", how="left")
         .fill_null(0))
    P = pl.col("tp") / pl.col("np")
    R = pl.col("tp") / pl.col("nt")
    f = (pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * P * R / (0.25 * P + R)))
    return d.select(f.mean()).item()


def decide(ij: pl.DataFrame, p: np.ndarray, params: dict) -> pl.DataFrame:
    """Return selected (i, j) pairs."""
    d = ij.select("i", "j").with_columns(p=pl.Series(np.clip(p * params.get("scale", 1.0), 0, 1), dtype=pl.Float64))
    if params.get("one_to_one", True):
        d = d.filter(pl.col("p") >= pl.col("p").max().over("j") - 1e-9)
    if params["mode"] == "thr":
        return d.filter(pl.col("p") >= params["floor"]).select("i", "j")
    # expected-F0.5 maximisation per S1 (candidates as independent Bernoullis)
    d = d.with_columns(
        tot=pl.col("p").sum().over("i"),
        p_empty=(1 - pl.col("p") + 1e-9).log().sum().over("i").exp(),
    ).filter(pl.col("p") >= params["floor"]).sort(["i", "p"], descending=[False, True])
    d = d.with_columns(k=pl.col("p").cum_count().over("i"), tp=pl.col("p").cum_sum().over("i"))
    d = d.with_columns(ef=1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("tot") - pl.col("tp"))
                                                  + (pl.col("k") - pl.col("tp"))))
    d = d.with_columns(best=pl.col("ef").max().over("i"))
    kbest = d.filter(pl.col("ef") == pl.col("best")).group_by("i").agg(pl.col("k").min().alias("kb"))
    d = d.join(kbest, on="i").filter((pl.col("k") <= pl.col("kb")) & (pl.col("best") > pl.col("p_empty")))
    return d.select("i", "j")


def tune_decision(ij, p, truth, s1_idx):
    best = None
    grid = [dict(mode="thr", floor=float(t), one_to_one=o) for o in (True, False)
            for t in np.round(np.arange(0.15, 0.96, 0.02), 3)]
    grid += [dict(mode="ef", floor=f, one_to_one=o, scale=sc) for o in (True, False)
             for f in (0.01, 0.05, 0.1, 0.2, 0.3) for sc in (0.9, 1.0, 1.1)]
    for prm in grid:
        sc = macro_f05(decide(ij, p, prm), truth, s1_idx)
        if best is None or sc > best[0]:
            best = (sc, prm)
    return best


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def make_model(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=600, max_leaf_nodes=127, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.08,
        n_iter_no_change=30, random_state=seed)


def fit_oof(parts, gpath, extra, y, fold, train_rows, K, seed, cols=None):
    """Train K fold models on sampled rows; predict every row out-of-fold."""
    Xs = pl.concat([X for _, _, X in iter_X(parts, gpath, extra, rows=train_rows)])
    cols = cols or Xs.columns
    Xs = Xs.select(cols).to_numpy()
    ys, fs = y[train_rows], fold[train_rows]
    log(f"  training rows {len(ys)} x {Xs.shape[1]} features, positives {ys.mean():.3f}")
    models = []
    for k in range(K):
        m = make_model(seed + k).fit(Xs[fs != k], ys[fs != k])
        models.append(m)
        log(f"  fold {k}: {m.n_iter_} iters")
    del Xs
    oof = np.zeros(len(y), dtype=np.float32)
    for off, n, X in iter_X(parts, gpath, extra):
        Xn = X.select(cols).to_numpy()
        fk = fold[off:off + n]
        for k in range(K):
            m = fk == k
            if m.any():
                oof[off:off + n][m] = models[k].predict_proba(Xn[m])[:, 1]
    return oof, models, cols


def predict_avg(models, parts, gpath, extra, cols):
    out = []
    for _, _, X in iter_X(parts, gpath, extra):
        Xn = X.select(cols).to_numpy()
        out.append(np.mean([m.predict_proba(Xn)[:, 1] for m in models], axis=0))
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def load_truth(data_dir, s1, tg):
    gt = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), separator="\t",
                     infer_schema=False).fill_null("")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids").str.strip_chars() != "")
            .join(s1.select("id").with_row_index("i"), left_on="source1_entity_id", right_on="id")
            .join(tg.select("id").with_row_index("j"),
                  left_on=pl.col("matched_entity_ids").str.strip_chars(), right_on="id")
            .select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32)))


def cmd_train(a):
    work = a.work_dir
    s1, tg = load_split(a.data_dir, "train", os.path.join(work, "norm"), a.workers)
    log(f"train records: S1={s1.height} S2+S3={tg.height}")
    truth = load_truth(a.data_dir, s1, tg)
    multi = truth.group_by("j").len().filter(pl.col("len") > 1).height
    log(f"truth links={truth.height}, targets linked to >1 S1: {multi}")
    cand = candidates("train", s1, tg, work, a.block_cfg)
    log(f"candidates: {cand.height} ({cand.height / s1.height:.1f} per S1), "
        f"blocking recall={blocking_recall(cand, truth):.4f}")
    all_i = pl.Series(np.arange(s1.height, dtype=np.uint32))
    log(f"oracle macro-F0.5 given candidates: "
        f"{macro_f05(cand.join(truth, on=['i', 'j'], how='semi'), truth, all_i):.4f}")
    y = cand.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["i", "j"], how="left")["y"] \
        .fill_null(0).to_numpy()
    parts, gpath = feature_parts("train", s1, tg, cand, work)
    log("features ready")

    h = s1.select(pl.col("id").hash(seed=7)).to_series().to_numpy()
    fold_s1 = (h % a.folds).astype(np.int8)
    samp_s1 = ((h // a.folds) % 1000) < int(a.sample * 1000)
    ci = cand["i"].to_numpy()
    fold, train_rows = fold_s1[ci], samp_s1[ci]

    oof1, m1, cols1 = fit_oof(parts, gpath, None, y, fold, train_rows, a.folds, seed=0)
    log("stage-1 done")
    C = context_features(cand, oof1)
    oof2, m2, cols2 = fit_oof(parts, gpath, C, y, fold, train_rows, a.folds, seed=100)
    log("stage-2 done")

    # tune the decision on held-out S1 (not used for fitting) -- all their competitors are scored OOF
    eval_i = pl.Series(np.where(~samp_s1)[0].astype(np.uint32))
    if len(eval_i) > a.eval_s1:
        eval_i = eval_i.sample(a.eval_s1, seed=1)
    sub = pl.DataFrame({"i": eval_i})
    # restrict to eval S1 candidates + every competing pair for their targets (needed by one-to-one)
    tj = cand.join(sub, on="i", how="semi").select("j").unique()
    keep = cand.with_row_index("r").join(tj, on="j", how="semi")["r"].to_numpy()
    ev_cand, ev_truth = cand[keep], truth.join(sub, on="i", how="semi")
    res = {}
    for name, p in (("stage1", oof1), ("stage2", oof2)):
        sc, prm = tune_decision(ev_cand, p[keep], ev_truth, eval_i)
        res[name] = {"f05": sc, "params": prm}
        log(f"{name}: OOF macro-F0.5={sc:.4f} with {prm}")
    stage = "stage2" if res["stage2"]["f05"] >= res["stage1"]["f05"] else "stage1"
    pred = decide(cand, oof2 if stage == "stage2" else oof1, res[stage]["params"])
    for c in sorted(s1["country"].unique()):
        ids = pl.Series(np.where(s1["country"].to_numpy() == c)[0].astype(np.uint32))
        log(f"  country={c!r}: F0.5={macro_f05(pred, truth, ids):.4f}")
    sing = set(range(s1.height)) - set(truth["i"].unique().to_list())
    sing = pl.Series(sorted(sing), dtype=pl.UInt32)
    log(f"  singletons F0.5={macro_f05(pred, truth, sing):.4f} (n={len(sing)})")

    joblib.dump(dict(m1=m1, m2=m2, cols1=cols1, cols2=cols2, stage=stage, params=res[stage]["params"],
                     block_cfg=a.block_cfg), os.path.join(work, "model.joblib"))
    with open(os.path.join(work, "cv_report.json"), "w") as f:
        json.dump(res | {"chosen": stage}, f, indent=2)
    log(f"saved model to {work}")


def write_lists(path, s1_ids, pairs: pl.DataFrame, tg_ids, col):
    lists = (pairs.with_columns(pl.Series("tid", tg_ids)[pairs["j"]])
             .group_by("i").agg(pl.col("tid").unique().sort().str.join(",")))
    out = (pl.DataFrame({"source1_entity_id": s1_ids}).with_row_index("i")
           .with_columns(pl.col("i").cast(pl.UInt32))
           .join(lists, on="i", how="left").sort("i")
           .select("source1_entity_id", pl.col("tid").fill_null("").alias(col)))
    out.write_csv(path, separator="\t", quote_style="never")


def cmd_predict(a):
    work = a.work_dir
    b = joblib.load(os.path.join(work, "model.joblib"))
    s1, tg = load_split(a.data_dir, "test", os.path.join(work, "norm"), a.workers)
    log(f"test records: S1={s1.height} S2+S3={tg.height}")
    cand = candidates("test", s1, tg, work, b["block_cfg"])
    log(f"candidates: {cand.height} ({cand.height / s1.height:.1f} per S1)")
    parts, gpath = feature_parts("test", s1, tg, cand, work)
    p = predict_avg(b["m1"], parts, gpath, None, b["cols1"])
    if b["stage"] == "stage2":
        p = predict_avg(b["m2"], parts, gpath, context_features(cand, p), b["cols2"])
    pred = decide(cand, p, b["params"])
    os.makedirs(a.out_dir, exist_ok=True)
    s1_ids, tg_ids = s1["id"].to_list(), tg["id"].to_list()
    write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, pred, tg_ids, "matched_entity_ids")
    write_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1_ids, cand, tg_ids, "candidate_entity_ids")
    if a.dump_scores:
        cand.select("i", "j").with_columns(p=pl.Series(p)).write_parquet(os.path.join(work, "test_scores.parquet"))
    matched = pred["i"].n_unique()
    log(f"wrote {a.out_dir}: {matched}/{s1.height} S1 with matches, {pred.height} links")
    for c in sorted(s1["country"].unique()):
        idx = np.where(s1["country"].to_numpy() == c)[0]
        rate = pred.filter(pl.col("i").is_in(idx.astype(np.uint32))).select("i").n_unique() / len(idx)
        log(f"  country={c!r}: n={len(idx)} matched-rate={rate:.3f}")
    assert pred.join(cand, on=["i", "j"], how="anti").height == 0, "match outside candidates"
    log("self-check PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict", "run"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work", help="cache for normalised data, candidates, features, model")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--sample", type=float, default=0.1, help="fraction of S1 entities used to fit models")
    ap.add_argument("--eval-s1", type=int, default=300_000, help="held-out S1 used to tune the decision")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--block-cfg", type=json.loads, default=None, help='JSON, e.g. {"max_per_s1":40}')
    ap.add_argument("--dump-scores", action="store_true")
    a = ap.parse_args()
    a.block_cfg = {**DEFAULT_BLOCKING, **(a.block_cfg or {})}
    if a.command in ("train", "run"):
        cmd_train(a)
    if a.command in ("predict", "run"):
        cmd_predict(a)


if __name__ == "__main__":
    main()
