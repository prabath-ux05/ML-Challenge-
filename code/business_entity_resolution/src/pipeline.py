"""End-to-end entity-resolution pipeline (scales to the full 2M x 10M challenge data).

    python src/pipeline.py train   --data-dir dataset --work-dir work
    python src/pipeline.py predict --data-dir dataset --work-dir work --out-dir output
    python src/pipeline.py run     ...   (train + predict)

Stages: normalise (parallel, cached) -> key blocking -> pair features (chunked parquet)
-> group features -> stage-1 GBDT (K folds, OOF on every train pair) -> context features
-> stage-2 GBDT -> F0.5-optimal decision (tuned on OOF) -> TSV outputs.
"""
import argparse
import gc
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
                      pair_features, token_idf)

FEATURE_VERSION = 4  # bump when pair/group features change, so cached parts are rebuilt
from expand import DEFAULT_EXPANSION, expansion_candidates  # noqa: E402
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


def feature_parts(split, s1, tg, cand, work, chunk=2_000_000, group=True):
    """Pair features written as row-aligned parquet parts (+ optionally one group-feature file)."""
    fdir = os.path.join(work, f"{split}_feat_v{FEATURE_VERSION}")
    done = os.path.join(fdir, "_DONE")
    if not os.path.exists(done):
        os.makedirs(fdir, exist_ok=True)
        f1, ft = name_frequencies(s1, tg)
        nidf = token_idf(s1, tg, "name_core", r"^[a-z0-9]{2,}$")
        aidf = token_idf(s1, tg, "addr_core", r"^[a-z]{2,}$")
        slim = []
        for k, s in enumerate(range(0, cand.height, chunk)):
            path = os.path.join(fdir, f"part_{k:04d}.parquet")
            try:  # resume: reuse parts finished by an interrupted run
                F = pl.read_parquet(path, columns=GROUP_BASE)
                assert F.height == min(chunk, cand.height - s)
                slim.append(F)
                continue
            except Exception:
                pass
            F = pair_features(s1, tg, cand.slice(s, chunk), f1, ft, nidf, aidf)
            F.write_parquet(path + ".tmp")
            os.replace(path + ".tmp", path)  # atomic: a killed run never leaves a half-written part
            slim.append(F.select([c for c in GROUP_BASE if c in F.columns]))
            del F
            gc.collect()
            log(f"  features {split}: {min(s + chunk, cand.height)}/{cand.height}")
        if group:
            slim = pl.concat([cand.select("i", "j"), pl.concat(slim)], how="horizontal")
            group_features(slim).write_parquet(os.path.join(fdir, "group.parquet"), row_group_size=500_000)
        del slim
        gc.collect()
        open(done, "w").close()
    return sorted(glob.glob(os.path.join(fdir, "part_*.parquet"))), os.path.join(fdir, "group.parquet")


def union_group(split, cand_u, parts_u, work):
    """Group (rank/gap) features recomputed over base + expansion candidates."""
    path = os.path.join(work, f"{split}_group_union_v{FEATURE_VERSION}.parquet")
    if not os.path.exists(path):
        slim = pl.concat([pl.read_parquet(p, columns=GROUP_BASE) for p in parts_u])
        slim = pl.concat([cand_u.select("i", "j"), slim], how="horizontal")
        group_features(slim).write_parquet(path + ".tmp", row_group_size=500_000)
        os.replace(path + ".tmp", path)
    return path


def _slice(path_or_df, off, n):
    if path_or_df is None:
        return None
    if isinstance(path_or_df, str):  # parquet on disk: read only the needed rows
        return pl.scan_parquet(path_or_df).slice(off, n).collect()
    return path_or_df.slice(off, n)


def iter_X(parts, group_path, extra=None, rows=None, goff=0):
    """Yield (offset, n, X) chunks = pair features + group features (+ extra columns).

    group_path / extra may be parquet paths (sliced lazily) or in-memory frames.
    goff: row offset of `parts` inside the group file (for the expansion rows of a union file)."""
    off = 0
    for p in parts:
        F = pl.read_parquet(p)
        n = F.height
        blocks = [F, _slice(group_path, goff + off, n)]
        if extra is not None:
            blocks.append(_slice(extra, off, n))
        X = pl.concat(blocks, how="horizontal")
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


SEG_FLAGS = ["seg_exp", "seg_noaddr", "seg_s3"]


def segment_flags(cand_u: pl.DataFrame, tg: pl.DataFrame, nb: int) -> pl.DataFrame:
    """Per-pair segment flags used for segment-specific score scaling in the decision."""
    t = tg.select(noaddr=pl.col("addr_core").str.len_chars() < 3, s3=pl.col("id").str.starts_with("S3"))
    j = cand_u["j"]
    return pl.DataFrame({
        "seg_exp": np.arange(cand_u.height) >= nb,
        "seg_noaddr": t["noaddr"].gather(j).to_numpy(),
        "seg_s3": t["s3"].gather(j).to_numpy(),
    })


def _adjust(p: np.ndarray, params: dict, seg: pl.DataFrame | None) -> np.ndarray:
    q = p * params.get("scale", 1.0)
    if seg is not None:
        for f in SEG_FLAGS:
            k = params.get(f, 1.0)
            if k != 1.0:
                q = np.where(seg[f].to_numpy(), q * k, q)
    return np.clip(q, 0, 1)


def decide(ij: pl.DataFrame, p: np.ndarray, params: dict, seg: pl.DataFrame | None = None,
           has_match: pl.DataFrame | None = None) -> pl.DataFrame:
    """Return selected (i, j) pairs.

    seg: per-pair segment flags (see segment_flags); params may scale each segment's scores.
    has_match: optional (i, q) entity-level P(any match) from the singleton model; blended into
    the probability of the empty prediction with weight params["w_single"]."""
    d = ij.select("i", "j").with_columns(p=pl.Series(_adjust(p, params, seg), dtype=pl.Float64))
    if params.get("one_to_one", True):
        d = d.filter(pl.col("p") >= pl.col("p").max().over("j") - 1e-9)
    if params["mode"] == "thr":
        return d.filter(pl.col("p") >= params["floor"]).select("i", "j")
    # expected-F0.5 maximisation per S1 (candidates as independent Bernoullis)
    d = d.with_columns(
        tot=pl.col("p").sum().over("i"),
        p_empty=(1 - pl.col("p") + 1e-9).log().sum().over("i").exp(),
    )
    w = params.get("w_single", 0.0)
    if has_match is not None and w > 0:
        d = d.join(has_match, on="i", how="left").with_columns(
            p_empty=(1 - w) * pl.col("p_empty") + w * (1 - pl.col("q").fill_null(0.0))).drop("q")
    d = d.filter(pl.col("p") >= params["floor"]).sort(["i", "p"], descending=[False, True])
    d = d.with_columns(k=pl.col("p").cum_count().over("i"), tp=pl.col("p").cum_sum().over("i"))
    d = d.with_columns(ef=1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("tot") - pl.col("tp"))
                                                  + (pl.col("k") - pl.col("tp"))))
    d = d.with_columns(best=pl.col("ef").max().over("i"))
    kbest = d.filter(pl.col("ef") == pl.col("best")).group_by("i").agg(pl.col("k").min().alias("kb"))
    d = d.join(kbest, on="i").filter((pl.col("k") <= pl.col("kb")) & (pl.col("best") > pl.col("p_empty")))
    return d.select("i", "j")


def tune_decision(ij, p, truth, s1_idx, seg=None, has_match=None):
    """Grid over global rules, then coordinate search over segment scales and singleton weight."""
    def score(prm):
        return macro_f05(decide(ij, p, prm, seg, has_match), truth, s1_idx)

    best = None
    grid = [dict(mode="thr", floor=float(t), one_to_one=o) for o in (True, False)
            for t in np.round(np.arange(0.15, 0.96, 0.02), 3)]
    grid += [dict(mode="ef", floor=f, one_to_one=o, scale=sc) for o in (True, False)
             for f in (0.01, 0.05, 0.1, 0.2, 0.3) for sc in (0.8, 0.9, 1.0, 1.1)]
    for prm in grid:
        sc = score(prm)
        if best is None or sc > best[0]:
            best = (sc, prm)
    extra = [(f, (0.7, 0.8, 0.9, 1.1, 1.2)) for f in SEG_FLAGS if seg is not None]
    if has_match is not None and best[1]["mode"] == "ef":
        extra.append(("w_single", (0.25, 0.5, 0.75, 1.0)))
    for _ in range(2):  # two passes of coordinate search
        for key, values in extra:
            for v in values:
                prm = {**best[1], key: v}
                sc = score(prm)
                if sc > best[0] + 1e-6:
                    best = (sc, prm)
    return best


def singleton_model(ij: pl.DataFrame, p: np.ndarray, truth: pl.DataFrame | None, s1: pl.DataFrame,
                    fold_s1=None, models=None, K=4):
    """Entity-level P(S1 has any match among its candidates) from its score profile.

    Train mode (truth given): fits K fold models, returns (has_match_oof, models).
    Predict mode: returns (has_match, models) averaged over the given models."""
    g = ij.select("i").with_columns(p=pl.Series(p, dtype=pl.Float32))
    s = pl.col("p").sort(descending=True)
    E = g.group_by("i").agg(
        p_max=s.first(), p_2nd=s.get(1, null_on_oob=True).fill_null(0), p_3rd=s.get(2, null_on_oob=True).fill_null(0),
        p_sum=pl.col("p").sum(), n50=(pl.col("p") > 0.5).sum(), n20=(pl.col("p") > 0.2).sum(), n=pl.len(),
    ).sort("i")
    nf = s1.select(pl.col("name_core")).with_row_index("i").join(
        s1.group_by("name_core").agg(pl.len().alias("name_freq")), on="name_core", how="left").select("i", "name_freq")
    E = E.join(nf.with_columns(pl.col("i").cast(pl.UInt32)), on="i", how="left")
    X = E.drop("i").to_numpy().astype(np.float32)
    ii = E["i"].to_numpy()
    if truth is not None:
        y = np.isin(ii, truth["i"].unique().to_numpy()).astype(np.int8)
        q = np.zeros(len(ii), dtype=np.float32)
        models = []
        fk = fold_s1[ii]
        for k in range(K):
            m = (make_model(k) if HAVE_LGB else
                 HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=k))
            m.fit(X[fk != k], y[fk != k])
            q[fk == k] = m.predict_proba(X[fk == k])[:, 1]
            models.append(m)
    else:
        q = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0).astype(np.float32)
    return pl.DataFrame({"i": E["i"], "q": q}), models


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
try:  # LightGBM (MIT) when its OpenMP runtime is available; otherwise sklearn's histogram GBDT
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:  # ImportError, or OSError when libomp is missing
    HAVE_LGB = False


class LGBModel:
    """LightGBM booster with early stopping on a random 8% hold-out; sklearn-like predict_proba."""

    def __init__(self, seed):
        self.seed = seed
        self.params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50,
                           feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                           max_bin=255, verbose=-1, seed=seed, num_threads=os.cpu_count() or 4)

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        va = rng.random(len(y)) < 0.08
        w = sample_weight if sample_weight is not None else np.ones(len(y), np.float32)
        dtr = lgb.Dataset(X[~va], y[~va], weight=w[~va], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], weight=w[va], reference=dtr)
        self.booster = lgb.train(self.params, dtr, num_boost_round=4000, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(100, verbose=False)])
        self.n_iter_ = self.booster.best_iteration
        return self

    def predict_proba(self, X):
        p = self.booster.predict(X, num_iteration=self.booster.best_iteration)
        return np.c_[1 - p, p]


class _Booster:
    """Wraps a trained lgb.Booster with the predict_proba interface used by predict_oof/predict_avg."""

    def __init__(self, booster):
        self.booster, self.n_iter_ = booster, booster.best_iteration

    def predict_proba(self, X):
        p = self.booster.predict(X, num_iteration=self.booster.best_iteration)
        return np.c_[1 - p, p]


def _fit_lgb_folds(X, y, w, fold, K, seed):
    base = lgb.Dataset(X, y, weight=w, params={"max_bin": 255, "verbose": -1}, free_raw_data=True).construct()
    rng = np.random.default_rng(seed)
    va_all = rng.random(len(y)) < 0.06  # early-stopping hold-out, disjoint from each fold's training rows
    prm = LGBModel(seed).params
    models = []
    for k in range(K):
        tr = np.where((fold != k) & ~va_all)[0]
        va = np.where((fold != k) & va_all)[0]
        dtr, dva = base.subset(tr), base.subset(va)
        b = lgb.train({**prm, "seed": seed + k}, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False, min_delta=2e-5)])
        models.append(_Booster(b))
        log(f"  fold {k}: {b.best_iteration} iters")
        del dtr, dva
        gc.collect()
    return models


def make_model(seed, kind="auto"):
    if kind == "lgb" or (kind == "auto" and HAVE_LGB):
        return LGBModel(seed)
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=2000, max_leaf_nodes=127, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.08,
        n_iter_no_change=50, random_state=seed)


def fit_oof(parts, gpath, extra, y, fold, train_rows, K, seed, cols=None):
    """Train K fold models on sampled rows; predict every row out-of-fold.

    train_rows: float weights per row (0 = not used; >0 = sample weight, e.g. 1/keep-rate for
    down-sampled negatives). The design matrix is preallocated to keep peak memory flat."""
    use = train_rows > 0
    n = int(use.sum())
    X = None
    pos = 0
    for off, m, C in iter_X(parts, gpath, extra, rows=use):
        cols = cols or C.columns
        if X is None:
            X = np.empty((n, len(cols)), dtype=np.float32)
        X[pos:pos + C.height] = C.select(cols).to_numpy()
        pos += C.height
        del C
    ys, fs, ws = y[use], fold[use], train_rows[use]
    log(f"  training rows {n} x {len(cols)} features, positives {ys.mean():.3f}, "
        f"model={'lightgbm' if HAVE_LGB else 'sklearn-hgb'}")
    if HAVE_LGB:  # bin once, then slice folds from the binned data (raw matrix freed immediately)
        models = _fit_lgb_folds(X, ys, ws, fs, K, seed)
        del X
        gc.collect()
        return predict_oof(models, parts, gpath, extra, cols, fold), models, cols
    models = []
    for k in range(K):
        tr = fs != k
        m = make_model(seed + k).fit(X[tr], ys[tr], sample_weight=ws[tr])
        models.append(m)
        log(f"  fold {k}: {m.n_iter_} iters")
        gc.collect()
    del X
    gc.collect()
    return predict_oof(models, parts, gpath, extra, cols, fold), models, cols


def predict_oof(models, parts, gpath, extra, cols, fold, goff=0):
    """Score rows with the model of their fold (models never saw those S1 entities)."""
    out = np.zeros(len(fold), dtype=np.float32)
    for off, n, X in iter_X(parts, gpath, extra, goff=goff):
        Xn = X.select(cols).to_numpy()
        fk, seg = fold[off:off + n], out[off:off + n]
        for k, m in enumerate(models):
            sel = fk == k
            if sel.any():
                seg[sel] = m.predict_proba(Xn[sel])[:, 1]
    return out


def predict_avg(models, parts, gpath, extra, cols, goff=0):
    out = []
    for _, _, X in iter_X(parts, gpath, extra, goff=goff):
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


SIB_COLS = ["sib_sn", "sib_sa", "sib_p", "sib_n"]


def _tag(path):
    st = os.stat(path)
    return f"{int(st.st_mtime)}_{st.st_size % 100000}"


def expand_union(split, s1, tg, cand, p1, parts_b, work, exp_cfg, tag):
    """Cluster expansion + features for the new pairs + group features over the union.

    Returns (cand_u, parts_e, union_group_path, sib) where sib holds sibling evidence
    (null for base rows) and an is_exp flag, aligned with cand_u."""
    name = f"{split}_exp_{tag}"
    path = os.path.join(work, f"{name}_cand.parquet")
    if os.path.exists(path):
        exp = pl.read_parquet(path)
    else:
        exp = expansion_candidates(s1, tg, cand, p1, exp_cfg, log=log)
        exp.write_parquet(path)
    log(f"expansion: +{exp.height} candidate pairs ({exp.height / s1.height:.2f} per S1)")
    ecand = exp.select("i", "j").with_columns(nkeys=pl.lit(0, pl.UInt8), keymask=pl.lit(0, pl.UInt32),
                                              bscore=pl.lit(0.0, pl.Float32))
    parts_e, _ = feature_parts(name, s1, tg, ecand, work, group=False)
    cand_u = pl.concat([cand.select("i", "j"), exp.select("i", "j")])
    g_u = union_group(name, cand_u, parts_b + parts_e, work)
    nb = cand.height
    base_sib = pl.select([pl.repeat(None, nb, dtype=pl.Float32).alias(c) for c in SIB_COLS])
    sib = pl.concat([base_sib, exp.select(SIB_COLS)]).with_columns(
        is_exp=pl.Series(np.r_[np.zeros(nb, np.float32), np.ones(exp.height, np.float32)]))
    return cand_u, parts_e, g_u, sib


def write_extra(path, cand_u, p1_u, tg, sib):
    """Stage-2 extra columns (context + sibling evidence), written to disk and sliced lazily."""
    if not os.path.exists(path):
        X = pl.concat([context_features(cand_u, p1_u, tg), sib], how="horizontal")
        X.write_parquet(path + ".tmp", row_group_size=500_000)
        os.replace(path + ".tmp", path)
        del X
        gc.collect()
    return path


def labels(cand, truth):
    return cand.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["i", "j"], how="left")["y"] \
        .fill_null(0).to_numpy()


def load_data(a, split):
    """Normalised (s1, tg); fast mode keeps a stable hash-sample of Source 1 entities."""
    s1, tg = load_split(a.data_dir, split, a.norm_dir or os.path.join(a.work_dir, "norm"), a.workers)
    if split == "train" and a.s1_frac < 1.0:
        s1 = s1.filter((pl.col("id").hash(seed=11) % 1000) < int(a.s1_frac * 1000))
        log(f"fast mode: keeping {s1.height} S1 entities ({a.s1_frac:.0%})")
    return s1, tg


def stratified_group_folds(s1: pl.DataFrame, truth: pl.DataFrame, K: int) -> np.ndarray:
    """Fold per S1 entity (the group): entities are shuffled by a stable hash and dealt round-robin
    within each stratum (country x has-any-match), so every fold has the same country mix and
    singleton rate. All candidate pairs of an entity share its fold -> no entity leakage."""
    has = np.zeros(s1.height, bool)
    has[truth["i"].unique().to_numpy()] = True
    d = (s1.select("country", h=pl.col("id").hash(seed=7)).with_row_index("i")
         .with_columns(has=pl.Series(has))
         .sort("country", "has", "h")
         .with_columns(fold=(pl.col("i").cum_count().over("country", "has") % K).cast(pl.Int8))
         .sort("i"))
    return d["fold"].to_numpy()


def append_experiment(work: str, record: dict):
    """One JSON line per training run -> the experiment log for the methodology document."""
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(os.path.join(work, "experiments.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def read_cols(parts, cols):
    return pl.concat([pl.read_parquet(p, columns=cols) for p in parts]) if parts else pl.DataFrame()


def cmd_train(a):
    work = a.work_dir
    s1, tg = load_data(a, "train")
    log(f"train records: S1={s1.height} S2+S3={tg.height}")
    truth = load_truth(a.data_dir, s1, tg)
    multi = truth.group_by("j").len().filter(pl.col("len") > 1).height
    log(f"truth links={truth.height}, targets linked to >1 S1: {multi}")
    all_i = pl.Series(np.arange(s1.height, dtype=np.uint32))
    cand = candidates("train", s1, tg, work, a.block_cfg)
    log(f"candidates: {cand.height} ({cand.height / s1.height:.1f} per S1), "
        f"blocking recall={blocking_recall(cand, truth):.4f}")
    log(f"oracle macro-F0.5 given candidates: "
        f"{macro_f05(cand.join(truth, on=['i', 'j'], how='semi'), truth, all_i):.4f}")
    y = labels(cand, truth)
    parts, gpath = feature_parts("train", s1, tg, cand, work)
    log("features ready")

    h = s1.select(pl.col("id").hash(seed=7)).to_series().to_numpy()
    fold_s1 = stratified_group_folds(s1, truth, a.folds)
    fit_ok = np.ones(s1.height, bool)
    if a.fit_countries:  # unseen-country proxy: fit only on these countries, score everything OOF
        fit_ok = s1["country"].is_in(a.fit_countries).to_numpy()
        log(f"fitting only on countries {a.fit_countries} ({fit_ok.sum()} S1)")
    samp_s1 = (((h // a.folds) % 1000) < int(a.sample * 1000)) & fit_ok
    rng = np.random.default_rng(0)

    def rows_for(ci, yy, parts_list):
        """Sampled S1 entities: all positives, all *hard* negatives (a similar name or address),
        and a random share of the easy negatives, re-weighted by 1/keep-rate."""
        H = read_cols(parts_list, ["n_tsort", "a_tset", "n_var_best"])
        hard = ((H["n_tsort"] >= 80) | (H["a_tset"] >= 80) | (H["n_var_best"] >= 80)).to_numpy()
        keep_neg = hard | (rng.random(len(yy)) < a.neg_keep)
        w_neg = np.where(hard, 1.0, 1.0 / a.neg_keep)
        return np.where(samp_s1[ci] & ((yy == 1) | keep_neg), np.where(yy == 1, 1.0, w_neg), 0.0).astype(np.float32)

    ci = cand["i"].to_numpy()
    fold = fold_s1[ci]
    ck1 = os.path.join(work, "stage1.joblib")
    if os.path.exists(ck1):  # resume
        st1 = joblib.load(ck1)
        m1, cols1, oof1 = st1["m1"], st1["cols1"], st1["oof1"]
        log("stage-1 loaded from checkpoint")
    else:
        oof1, m1, cols1 = fit_oof(parts, gpath, None, y, fold, rows_for(ci, y, parts), a.folds, seed=0)
        joblib.dump(dict(m1=m1, cols1=cols1, oof1=oof1), ck1)
        log("stage-1 done")

    # cluster expansion (from out-of-fold stage-1 scores) -> union candidate set
    tag = _tag(ck1)
    cand_u, parts_e, g_u, sib = expand_union("train", s1, tg, cand, oof1, parts, work, a.exp_cfg, tag)
    nb = cand.height
    y_u = labels(cand_u, truth)
    log(f"union candidates: {cand_u.height}, recall={blocking_recall(cand_u, truth):.4f}, oracle macro-F0.5="
        f"{macro_f05(cand_u.join(truth, on=['i', 'j'], how='semi'), truth, all_i):.4f}")
    ci_u = cand_u["i"].to_numpy()
    fold_u = fold_s1[ci_u]
    p1_e = predict_oof(m1, parts_e, g_u, None, cols1, fold_u[nb:], goff=nb)
    p1_u = np.r_[oof1, p1_e].astype(np.float32)
    extra = write_extra(os.path.join(work, f"train_exp_{tag}_extra.parquet"), cand_u, p1_u, tg, sib)
    log("context features ready")
    oof2, m2, cols2 = fit_oof(parts + parts_e, g_u, extra, y_u, fold_u, rows_for(ci_u, y_u, parts + parts_e),
                              a.folds, seed=100)
    log("stage-2 done")
    cand_u.with_columns(y=pl.Series(y_u), p1=pl.Series(p1_u), p2=pl.Series(oof2),
                        is_exp=pl.Series(np.arange(cand_u.height) >= nb)) \
        .write_parquet(os.path.join(work, "train_oof.parquet"))

    # entity-level "has any match" model (for singletons), trained OOF on the stage-2 score profile
    truth_in = cand_u.filter(pl.Series(y_u == 1)).select("i", "j")
    has_oof, sm = singleton_model(cand_u, oof2, truth_in, s1, fold_s1=fold_s1, K=a.folds)
    seg_u = segment_flags(cand_u, tg, nb)

    # tune the decision on held-out S1 (not used for fitting) -- all their competitors are scored OOF
    eval_i = pl.Series(np.where(~samp_s1 & fit_ok)[0].astype(np.uint32))
    if len(eval_i) > a.eval_s1:
        eval_i = eval_i.sample(a.eval_s1, seed=1)
    sub = pl.DataFrame({"i": eval_i})
    # restrict to eval S1 candidates + every competing pair for their targets (needed by one-to-one)
    tj = cand_u.join(sub, on="i", how="semi").select("j").unique()
    keep = cand_u.with_row_index("r").join(tj, on="j", how="semi")["r"].to_numpy()
    ev_cand, ev_truth, ev_seg = cand_u[keep], truth.join(sub, on="i", how="semi"), seg_u[keep]
    res = {}
    for name, p, hm in (("stage1", p1_u, None), ("stage2", oof2, has_oof)):
        sc, prm = tune_decision(ev_cand, p[keep], ev_truth, eval_i, seg=ev_seg, has_match=hm)
        res[name] = {"f05": sc, "params": prm}
        log(f"{name}: OOF macro-F0.5={sc:.4f} with {prm}")
    stage = "stage2" if res["stage2"]["f05"] >= res["stage1"]["f05"] else "stage1"
    use_sm = stage == "stage2"
    pred = decide(cand_u, oof2 if stage == "stage2" else p1_u, res[stage]["params"], seg_u,
                  has_oof if use_sm else None)
    by_country = {}
    for c in sorted(s1["country"].unique()):
        ids = pl.Series(np.where(s1["country"].to_numpy() == c)[0].astype(np.uint32))
        by_country[c] = macro_f05(pred, truth, ids)
        log(f"  country={c!r}: F0.5={by_country[c]:.4f}")
    sing = pl.Series(np.setdiff1d(np.arange(s1.height), truth["i"].unique().to_numpy()).astype(np.uint32))
    f_sing = macro_f05(pred, truth, sing)
    log(f"  singletons F0.5={f_sing:.4f} (n={len(sing)})")
    append_experiment(work, dict(
        feature_version=FEATURE_VERSION, model="lightgbm" if HAVE_LGB else "sklearn-hgb", folds=a.folds,
        sample=a.sample, neg_keep=a.neg_keep, s1_frac=a.s1_frac, fit_countries=a.fit_countries,
        block_cfg=a.block_cfg, exp_cfg=a.exp_cfg, n_s1=s1.height, n_cand=cand.height, n_union=cand_u.height,
        recall_blocking=blocking_recall(cand, truth), recall_union=blocking_recall(cand_u, truth),
        oof_stage1=res["stage1"]["f05"], oof_stage2=res["stage2"]["f05"], chosen=stage,
        params=res[stage]["params"], by_country=by_country, singletons=f_sing))

    joblib.dump(dict(m1=m1, m2=m2, cols1=cols1, cols2=cols2, stage=stage, params=res[stage]["params"],
                     sm=sm if use_sm else None, block_cfg=a.block_cfg, exp_cfg=a.exp_cfg),
                os.path.join(work, "model.joblib"))
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
    mpath = os.path.join(work, "model.joblib")
    b = joblib.load(mpath)
    s1, tg = load_data(a, "test")
    log(f"test records: S1={s1.height} S2+S3={tg.height}")
    cand = candidates("test", s1, tg, work, b["block_cfg"])
    log(f"candidates: {cand.height} ({cand.height / s1.height:.1f} per S1)")
    parts, gpath = feature_parts("test", s1, tg, cand, work)
    p1 = predict_avg(b["m1"], parts, gpath, None, b["cols1"])
    tag = _tag(mpath)
    cand_u, parts_e, g_u, sib = expand_union("test", s1, tg, cand, p1, parts, work,
                                             b.get("exp_cfg", DEFAULT_EXPANSION), tag)
    nb = cand.height
    p1_e = predict_avg(b["m1"], parts_e, g_u, None, b["cols1"], goff=nb) if parts_e else np.zeros(0, np.float32)
    p1_u = np.r_[p1, p1_e].astype(np.float32)
    p = p1_u
    if b["stage"] == "stage2":
        extra = write_extra(os.path.join(work, f"test_exp_{tag}_extra.parquet"), cand_u, p1_u, tg, sib)
        p = predict_avg(b["m2"], parts + parts_e, g_u, extra, b["cols2"])
    seg_u = segment_flags(cand_u, tg, nb)
    has_match = singleton_model(cand_u, p, None, s1, models=b["sm"])[0] if b.get("sm") else None
    pred = decide(cand_u, p, b["params"], seg_u, has_match)
    os.makedirs(a.out_dir, exist_ok=True)
    s1_ids, tg_ids = s1["id"].to_list(), tg["id"].to_list()
    write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, pred, tg_ids, "matched_entity_ids")
    write_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1_ids, cand_u, tg_ids, "candidate_entity_ids")
    # always keep test scores: re-tuning decisions / diagnostics never need a re-run
    pl.concat([cand_u, seg_u], how="horizontal").with_columns(p1=pl.Series(p1_u), p=pl.Series(p)) \
        .write_parquet(os.path.join(work, "test_scores.parquet"))
    matched = pred["i"].n_unique()
    log(f"wrote {a.out_dir}: {matched}/{s1.height} S1 with matches, {pred.height} links")
    for c in sorted(s1["country"].unique()):
        idx = np.where(s1["country"].to_numpy() == c)[0]
        rate = pred.filter(pl.col("i").is_in(idx.astype(np.uint32))).select("i").n_unique() / len(idx)
        log(f"  country={c!r}: n={len(idx)} matched-rate={rate:.3f}")
    assert pred.join(cand_u, on=["i", "j"], how="anti").height == 0, "match outside candidates"
    log("self-check PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict", "run"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work", help="cache for normalised data, candidates, features, model")
    ap.add_argument("--norm-dir", default=None, help="normalised-record cache (default: <work-dir>/norm)")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sample", type=float, default=0.3, help="fraction of S1 entities used to fit models")
    ap.add_argument("--neg-keep", type=float, default=0.25, help="share of easy negative pairs kept for fitting")
    ap.add_argument("--eval-s1", type=int, default=300_000, help="held-out S1 used to tune the decision")
    ap.add_argument("--s1-frac", type=float, default=1.0, help="fast experiments: use this share of train S1")
    ap.add_argument("--fit-countries", type=lambda x: [c.strip().lower() for c in x.split(",")], default=None,
                    help="unseen-country proxy, e.g. 'us': fit on these countries only, report all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--block-cfg", type=json.loads, default=None, help='JSON, e.g. {"max_per_s1":40}')
    ap.add_argument("--exp-cfg", type=json.loads, default=None, help='JSON, e.g. {"tau":0.4,"min_sim":85}')
    a = ap.parse_args()
    a.block_cfg = {**DEFAULT_BLOCKING, **(a.block_cfg or {})}
    a.exp_cfg = {**DEFAULT_EXPANSION, **(a.exp_cfg or {})}
    if a.command in ("train", "run"):
        cmd_train(a)
    if a.command in ("predict", "run"):
        cmd_predict(a)


if __name__ == "__main__":
    main()
