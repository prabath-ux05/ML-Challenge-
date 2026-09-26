"""Inspect out-of-fold errors of the last training run.

    python src/error_analysis.py --work-dir work [--n 25]

Needs work/train_oof.parquet, work/model.joblib and work/norm/train_*.parquet (written by `train`).
Applies the tuned decision rule to the OOF scores, then prints false merges (FP) and missed
matches (FN) with a breakdown by simple error categories and by country.
"""
import argparse
import os
import sys

import joblib
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import decide  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--n", type=int, default=25, help="examples to print per error type")
    a = ap.parse_args()
    b = joblib.load(os.path.join(a.work_dir, "model.joblib"))
    oof = pl.read_parquet(os.path.join(a.work_dir, "train_oof.parquet"))
    s1 = pl.read_parquet(os.path.join(a.work_dir, "norm", "train_s1.parquet")).with_row_index("i")
    tg = pl.read_parquet(os.path.join(a.work_dir, "norm", "train_tg.parquet")).with_row_index("j")
    p = oof["p2" if b["stage"] == "stage2" else "p1"].to_numpy()
    pred = decide(oof, p, b["params"]).with_columns(pred=pl.lit(1, pl.Int8))
    d = oof.with_columns(p=pl.Series(p)).join(pred, on=["i", "j"], how="left").with_columns(pl.col("pred").fill_null(0))
    fp = d.filter((pl.col("pred") == 1) & (pl.col("y") == 0))
    fn = d.filter((pl.col("pred") == 0) & (pl.col("y") == 1))
    print(f"pairs={d.height}  predicted={pred.height}  FP={fp.height}  FN(in candidates)={fn.height}")

    cols1 = ["i", "country", "name_core", "addr_core"]
    colst = ["j", "id", "name_core", "addr_core", "name_vars"]

    def show(df, title):
        x = (df.join(s1.select(cols1), on="i").join(tg.select(colst), on="j", suffix="_t")
             .with_columns(
                 addr_missing=pl.col("addr_core_t").str.len_chars() < 3,
                 name_eq=pl.col("name_core") == pl.col("name_core_t"),
                 native=pl.col("name_vars").str.contains("[^a-z0-9 |]") | pl.col("name_core_t").str.contains(
                     r"\b(pvt|ltd)\b"),
             ))
        print(f"\n=== {title}: {x.height} ===")
        print(x.group_by("country").len().sort("len", descending=True))
        print(x.select(pl.col("addr_missing").mean(), pl.col("name_eq").mean()))
        for r in x.sort("p", descending=title.startswith("FP")).head(a.n).iter_rows(named=True):
            print(f"p={r['p']:.2f} [{r['country']}] {r['name_core']} | {r['addr_core']}\n"
                  f"          {r['id']}: {r['name_core_t']} | {r['addr_core_t']}")

    show(fp, "FP (false merges, most confident first)")
    show(fn, "FN (missed matches, least confident first)")


if __name__ == "__main__":
    main()
