"""Test-time prior correction for decoy groups (label shift within segments).

Finding (see README): test decoys come in *groups* of 2-3 records sharing a shifted house number
next to the real business. In train such groups were mostly true copies (P(true) 0.68-0.98), so the
model learned "a group of look-alikes at another number = real". In test they are ~6x more frequent
per entity while exact / same-number copies are unchanged, i.e. the extra ones are decoys.

For every candidate pair we define a segment from the entity's look-alikes (name >= 70, address >= 85):
    country x {same number | shifted number, group size 1 / 2 / 3+} x {entity has a same-number look-alike}
and re-estimate the share of true pairs per segment on test, then shift each pair's odds:
    odds' = odds * [pi_te / (1 - pi_te)] / [pi_tr / (1 - pi_tr)]
pi_te is estimated two ways: count-based (true pairs per S1 in the segment as in train) and EM
(Saerens et al. 2002) on the model's own test scores. France uses the pooled train segments.

    python src/recalibrate.py --work-dir work --out-dir output_v31_recal [--method count|em|mean]
"""
import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import decide, singleton_model, write_lists  # noqa: E402

LOOK = (pl.col("n_tsort") >= 70) & (pl.col("a_tset") >= 85)


def union_features(work: str, split: str, exp_tag: str, cols) -> pl.DataFrame:
    """Pair features for base + expansion candidates, row-aligned with the union candidate set."""
    base = sorted(glob.glob(os.path.join(work, f"{split}_feat_v*", "part_*.parquet")))
    exp = sorted(glob.glob(os.path.join(work, f"{split}_exp_{exp_tag}_feat_v*", "part_*.parquet")))
    return pl.concat([pl.read_parquet(p, columns=cols) for p in base + exp])


def segments(ij: pl.DataFrame, F: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame) -> pl.Series:
    """Segment label per pair (see module docstring); '<country>:other' for pairs that are not look-alikes."""
    d = pl.concat([ij.select("i", "j"), F], how="horizontal_extend").with_row_index("r")
    d = d.join(s1.select("i", "country"), on="i").join(tg.select("j", "first_num"), on="j")
    lk = d.filter(LOOK)
    home = lk.filter(pl.col("a_num_eq") == 1).select("i").unique().with_columns(home=pl.lit(1))
    sh = lk.filter((pl.col("a_num_eq") == 0) & (pl.col("first_num") != "")) \
        .with_columns(g=pl.len().over("i", "first_num").clip(1, 3))
    lab = (pl.concat([
        lk.filter(pl.col("a_num_eq") == 1).select("r", "i", "country", kind=pl.lit("same")),
        sh.select("r", "i", "country", kind=pl.lit("shift") + pl.col("g").cast(pl.String)),
    ]).join(home, on="i", how="left")
        .with_columns(seg=pl.col("country") + ":" + pl.col("kind") + ":home" + pl.col("home").fill_null(0).cast(pl.String)))
    out = d.select("r", "country").join(lab.select("r", "seg"), on="r", how="left").sort("r")
    return out.select(pl.col("seg").fill_null(pl.col("country") + ":other"))["seg"]


def em_prior(p: np.ndarray, pi_tr: float, iters=100) -> float:
    pi = pi_tr
    for _ in range(iters):
        a = (pi / pi_tr) * p
        b = ((1 - pi) / (1 - pi_tr)) * (1 - p)
        new = float(np.mean(a / (a + b + 1e-12)))
        if abs(new - pi) < 1e-6:
            break
        pi = new
    return pi


def adjust(p: np.ndarray, seg: pl.Series, ratio: dict) -> np.ndarray:
    r = seg.replace_strict(ratio, default=1.0, return_dtype=pl.Float64).to_numpy()
    q = p.astype(np.float64)
    return (r * q / (r * q + (1 - q))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default=None, help="write a submission with the adjusted scores")
    ap.add_argument("--method", default="count", choices=["count", "em", "mean"])
    ap.add_argument("--min-pairs", type=int, default=2000, help="segments smaller than this are left alone")
    ap.add_argument("--train-only", action="store_true", help="only print the train segment table")
    ap.add_argument("--train-work-dir", default=None, help="where train_oof / train features live (default: --work-dir)")
    a = ap.parse_args()
    w, wt = a.work_dir, a.train_work_dir or a.work_dir
    tr_tag = os.path.basename(glob.glob(os.path.join(wt, "train_exp_*_cand.parquet"))[0])[len("train_exp_"):-len("_cand.parquet")]
    te_tag = os.path.basename(glob.glob(os.path.join(w, "test_exp_*_cand.parquet"))[0])[len("test_exp_"):-len("_cand.parquet")]
    cols = ["n_tsort", "a_tset", "a_num_eq"]
    norm = lambda f, c: pl.read_parquet(os.path.join(wt if f.startswith("train") else w, "norm", f),  # noqa: E731
                                        columns=c).with_row_index("idx")

    # ---- train: segment priors and calibration
    s1 = norm("train_s1.parquet", ["country"]).rename({"idx": "i"}).with_columns(pl.col("i").cast(pl.UInt32))
    tg = norm("train_tg.parquet", ["first_num"]).rename({"idx": "j"}).with_columns(pl.col("j").cast(pl.UInt32))
    oof = pl.read_parquet(os.path.join(wt, "train_oof.parquet"), columns=["i", "j", "y", "p2"])
    seg_tr = segments(oof, union_features(wt, "train", tr_tag, cols), s1, tg)
    n1 = s1.group_by("country").len("ns1")
    T = (oof.with_columns(seg=seg_tr).join(s1, on="i").group_by("seg", "country")
         .agg(n_tr=pl.len(), pi_tr=pl.col("y").mean(), p_tr=pl.col("p2").mean())
         .join(n1, on="country").with_columns(true_per_s1=pl.col("n_tr") * pl.col("pi_tr") / pl.col("ns1"),
                                              pairs_per_s1_tr=pl.col("n_tr") / pl.col("ns1"))
         .with_columns(key=pl.col("seg").str.split_exact(":", 1).struct.field("field_1")
                       + ":" + pl.col("seg").str.split(":").list.last()))
    del oof, seg_tr
    if a.train_only:
        pl.Config.set_tbl_rows(80), pl.Config.set_tbl_width_chars(250), pl.Config.set_float_precision(3)
        print(T.select("seg", "n_tr", "pairs_per_s1_tr", "pi_tr", "p_tr", "true_per_s1").sort("seg"))
        return

    # ---- test
    s1t = norm("test_s1.parquet", ["country", "name_core"]).rename({"idx": "i"}).with_columns(pl.col("i").cast(pl.UInt32))
    tgt = norm("test_tg.parquet", ["first_num"]).rename({"idx": "j"}).with_columns(pl.col("j").cast(pl.UInt32))
    ts = pl.read_parquet(os.path.join(w, "test_scores.parquet"))
    seg_te = segments(ts, union_features(w, "test", te_tag, cols), s1t, tgt)
    p = ts["p"].to_numpy()
    n1t = s1t.group_by("country").len("ns1")
    E = (ts.select("i", "p").with_columns(seg=seg_te).join(s1t.select("i", "country"), on="i")
         .group_by("seg", "country").agg(n_te=pl.len(), p_te=pl.col("p").mean())
         .join(n1t, on="country").with_columns(pairs_per_s1_te=pl.col("n_te") / pl.col("ns1"))
         .with_columns(key=pl.col("seg").str.split(":").list.slice(1).list.join(":")))
    # train reference per segment: same country, or the pooled US+India segment for unseen countries
    pooled = (T.group_by("key").agg(pi_tr=(pl.col("pi_tr") * pl.col("n_tr")).sum() / pl.col("n_tr").sum(),
                                   true_per_s1=(pl.col("true_per_s1") * pl.col("ns1")).sum() / pl.col("ns1").sum(),
                                   n_tr=pl.col("n_tr").sum()))
    E = E.join(T.select("seg", "pi_tr", "true_per_s1", "n_tr", "pairs_per_s1_tr", "p_tr"), on="seg", how="left")
    E = E.join(pooled.rename({"pi_tr": "pi_pool", "true_per_s1": "tps_pool", "n_tr": "n_pool"}), on="key", how="left")
    E = E.with_columns(pi_tr=pl.col("pi_tr").fill_null(pl.col("pi_pool")),
                       true_per_s1=pl.col("true_per_s1").fill_null(pl.col("tps_pool")),
                       n_tr=pl.col("n_tr").fill_null(pl.col("n_pool")))
    E = E.with_columns(pi_count=(pl.col("true_per_s1") / pl.col("pairs_per_s1_te")).clip(0.001, 0.999))
    segp = pl.DataFrame({"seg": seg_te, "p": p})
    em = {}
    for row in E.iter_rows(named=True):
        if row["seg"].endswith("other") or row["pi_tr"] is None:
            continue
        pp = segp.filter(pl.col("seg") == row["seg"])["p"].to_numpy()
        em[row["seg"]] = em_prior(pp, float(np.clip(row["pi_tr"], 1e-3, 1 - 1e-3)))
    E = E.with_columns(pi_em=pl.col("seg").replace_strict(em, default=None, return_dtype=pl.Float64))
    E = E.with_columns(pi_te=pl.when(pl.lit(a.method == "count")).then(pl.col("pi_count"))
                       .when(pl.lit(a.method == "em")).then(pl.col("pi_em"))
                       .otherwise((pl.col("pi_count") + pl.col("pi_em")) / 2))
    odds = lambda x: pl.col(x).clip(1e-3, 1 - 1e-3) / (1 - pl.col(x).clip(1e-3, 1 - 1e-3))  # noqa: E731
    E = E.with_columns(ratio=pl.when((pl.col("n_te") >= a.min_pairs) & (pl.col("n_tr") >= a.min_pairs)
                                     & ~pl.col("seg").str.ends_with("other") & pl.col("pi_te").is_not_null())
                       .then(odds("pi_te") / odds("pi_tr")).otherwise(1.0))
    pl.Config.set_tbl_rows(80), pl.Config.set_tbl_cols(20), pl.Config.set_tbl_width_chars(250), pl.Config.set_float_precision(3)
    print(E.select("seg", "pairs_per_s1_tr", "pairs_per_s1_te", "pi_tr", "p_tr", "p_te", "pi_count", "pi_em", "ratio")
          .sort("seg"))
    ratio = dict(zip(E["seg"].to_list(), E["ratio"].to_list()))
    json.dump({"method": a.method, "ratio": ratio}, open(os.path.join(w, f"recal_{a.method}.json"), "w"), indent=1)
    if not a.out_dir:
        return

    # ---- adjusted decision -> submission
    q = adjust(p, seg_te, ratio)
    b = joblib.load(os.path.join(w, "model.joblib"))
    seg_flags = ts.select("seg_exp", "seg_noaddr", "seg_s3")
    s1n = s1t.select("name_core")
    hm = singleton_model(ts.select("i", "j"), q, None, s1n, models=b["sm"])[0] if b.get("sm") else None
    pred = decide(ts.select("i", "j"), q, b["params"], seg_flags, hm)
    os.makedirs(a.out_dir, exist_ok=True)
    s1_ids = pl.read_parquet(os.path.join(w, "norm", "test_s1.parquet"), columns=["id"])["id"].to_list()
    tg_ids = pl.read_parquet(os.path.join(w, "norm", "test_tg.parquet"), columns=["id"])["id"].to_list()
    write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, pred, tg_ids, "matched_entity_ids")
    write_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1_ids, ts.select("i", "j"), tg_ids,
                "candidate_entity_ids")
    base = decide(ts.select("i", "j"), p, b["params"], seg_flags,
                  singleton_model(ts.select("i", "j"), p, None, s1n, models=b["sm"])[0] if b.get("sm") else None)
    c = s1t.select("i", "country")
    for name, pr in (("before", base), ("after", pred)):
        st = c.join(pr.group_by("i").len("n"), on="i", how="left").fill_null(0).group_by("country") \
            .agg(links=pl.col("n").mean(), empty=(pl.col("n") == 0).mean()).sort("country")
        print(name, st.to_dicts())
    print(f"wrote {a.out_dir}: {pred.height} links (was {base.height})")


if __name__ == "__main__":
    main()
