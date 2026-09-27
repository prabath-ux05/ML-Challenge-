"""Blend the test scores of two trained runs (average of probabilities on the union of candidates).

    python src/blend.py --work-a work --work-b work_v4 --out-dir output_blend [--w-b 0.5]

Pairs scored by only one run keep that run's probability. The decision parameters and the
singleton model of run B are used. Also prints the model's own expected F0.5 per country for A, B
and the blend (the self-estimate predicted v3.1's leaderboard score to within 0.004).
"""
import argparse
import os
import sys

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import decide, singleton_model, write_lists  # noqa: E402
from shift_report import profile  # noqa: E402


def ids(work, f):
    return pl.read_parquet(os.path.join(work, "norm", f), columns=["id"])["id"]


def scores(work):
    """Test scores keyed by (S1 id, target id) so runs with different normalisation line up."""
    s1, tg = ids(work, "test_s1.parquet"), ids(work, "test_tg.parquet")
    ts = pl.read_parquet(os.path.join(work, "test_scores.parquet"), columns=["i", "j", "p", "seg_exp", "seg_noaddr", "seg_s3"])
    return ts.with_columns(sid=s1.gather(ts["i"]), tid=tg.gather(ts["j"])).drop("i", "j")


def self_f(pairs_ij, p, pred, country):
    d = pairs_ij.with_columns(p=pl.Series(p))
    return profile(d.select("i", "j", "p"), pred, country).select("country", "n_s1", "links/s1", "pred_empty", "self_F05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-a", default="work")
    ap.add_argument("--work-b", default="work_v4")
    ap.add_argument("--w-b", type=float, default=0.5, help="weight of run B in the average")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    A, B = scores(a.work_a), scores(a.work_b)
    u = A.select("sid", "tid", pa="p").join(B.rename({"p": "pb"}), on=["sid", "tid"], how="full", coalesce=True)
    u = u.with_columns(p=pl.when(pl.col("pa").is_null()).then(pl.col("pb"))
                       .when(pl.col("pb").is_null()).then(pl.col("pa"))
                       .otherwise((1 - a.w_b) * pl.col("pa") + a.w_b * pl.col("pb")))
    u = u.with_columns([pl.col(c).fill_null(False) for c in ("seg_exp", "seg_noaddr", "seg_s3")])
    # index against run B's record order
    s1b, tgb = ids(a.work_b, "test_s1.parquet"), ids(a.work_b, "test_tg.parquet")
    si = pl.DataFrame({"sid": s1b, "i": np.arange(len(s1b), dtype=np.uint32)})
    tj = pl.DataFrame({"tid": tgb, "j": np.arange(len(tgb), dtype=np.uint32)})
    u = u.join(si, on="sid").join(tj, on="tid").sort("i", "j")
    b = joblib.load(os.path.join(a.work_b, "model.joblib"))
    s1n = pl.read_parquet(os.path.join(a.work_b, "norm", "test_s1.parquet"), columns=["name_core", "country"])
    country = s1n.select("country").with_row_index("i").with_columns(pl.col("i").cast(pl.UInt32))
    ij = u.select("i", "j")
    out = {}
    for name, col in (("A", "pa"), ("B", "pb"), ("blend", "p")):
        m = u.filter(pl.col(col).is_not_null())
        p = m[col].to_numpy()
        hm = singleton_model(m.select("i", "j"), p, None, s1n, models=b["sm"])[0] if b.get("sm") else None
        pred = decide(m.select("i", "j"), p, b["params"], m.select("seg_exp", "seg_noaddr", "seg_s3"), hm)
        rep = self_f(m.select("i", "j"), p, pred, country)
        tot = (rep["n_s1"] * rep["self_F05"]).sum() / rep["n_s1"].sum()
        print(f"--- {name}: model's own expected F0.5 = {tot:.4f}")
        print(rep)
        out[name] = pred
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1b.to_list(), out["blend"], tgb.to_list(),
                    "matched_entity_ids")
        write_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), s1b.to_list(), ij, tgb.to_list(),
                    "candidate_entity_ids")
        assert out["blend"].join(ij, on=["i", "j"], how="anti").height == 0
        print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
