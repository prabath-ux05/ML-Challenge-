"""Train (OOF) vs test score profiles per country: is the test set harder, or is the model miscalibrated?

    python src/shift_report.py --work-dir work

For every S1 entity: expected number of matches (sum of p), links chosen by the decision, the
model's own expected F0.5 for that choice, and how many pairs fall in each probability band.
On train the expected F0.5 is compared with the true one (calibration); on test it is the model's
self-estimate, to set against the leaderboard.
"""
import argparse
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import decide, load_truth, macro_f05  # noqa: E402

BANDS = [(0.9, 1.01), (0.5, 0.9), (0.2, 0.5), (0.05, 0.2)]


def profile(pairs: pl.DataFrame, pred: pl.DataFrame, country: pl.DataFrame,
            n_true: pl.DataFrame | None = None) -> pl.DataFrame:
    """pairs: (i, j, p[, y]); pred: chosen (i, j); n_true: (i, n_true) from the full ground truth
    (includes matches blocking missed). Returns one row of stats per country."""
    has_y = "y" in pairs.columns
    chosen = pred.with_columns(sel=pl.lit(True))
    d = pairs.join(chosen, on=["i", "j"], how="left").with_columns(pl.col("sel").fill_null(False))
    agg = [pl.len().alias("n_pairs"), pl.col("p").sum().alias("sum_p"),
           pl.col("sel").sum().alias("n_sel"), (pl.col("p") * pl.col("sel")).sum().alias("tp_exp"),
           (1 - pl.col("p") + 1e-9).log().sum().exp().alias("p_none")]
    agg += [((pl.col("p") >= lo) & (pl.col("p") < hi)).sum().alias(f"band_{lo}") for lo, hi in BANDS]
    if has_y:
        agg += [(pl.col("y") * pl.col("sel")).sum().alias("tp")]
    e = d.group_by("i").agg(agg)
    if has_y:
        e = e.join(n_true, on="i", how="left").with_columns(pl.col("n_true").fill_null(0))
    # model's own expected F0.5 of the chosen set (same ratio-of-expectations form as decide())
    ef = (pl.when(pl.col("n_sel") == 0).then(pl.col("p_none"))
          .otherwise(1.25 * pl.col("tp_exp") / (pl.col("n_sel") + 0.25 * pl.col("sum_p"))))
    e = e.with_columns(ef=ef)
    e = country.join(e, on="i", how="left").fill_null(0)  # S1 without candidates: predicted empty
    if has_y:  # re-attach n_true for S1 without candidates too
        e = e.drop("n_true").join(n_true, on="i", how="left").with_columns(pl.col("n_true").fill_null(0))
        P, R = pl.col("tp") / pl.col("n_sel"), pl.col("tp") / pl.col("n_true")
        e = e.with_columns(f=pl.when(pl.col("n_true") == 0).then((pl.col("n_sel") == 0).cast(pl.Float64))
                           .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * P * R / (0.25 * P + R)))
    e = e.with_columns(ef=pl.when(pl.col("n_pairs") == 0).then(1.0).otherwise(pl.col("ef")))
    out = [pl.len().alias("n_s1"), pl.col("n_pairs").mean().alias("pairs/s1"),
           pl.col("sum_p").mean().alias("E[matches]/s1"), pl.col("n_sel").mean().alias("links/s1"),
           (pl.col("n_sel") == 0).mean().alias("pred_empty"), pl.col("ef").mean().alias("self_F05")]
    out += [pl.col(f"band_{lo}").mean().alias(f"p{lo}-{min(hi, 1.0)}") for lo, hi in BANDS]
    if has_y:
        out += [pl.col("n_true").mean().alias("true/s1"), (pl.col("n_true") == 0).mean().alias("true_empty"),
                pl.col("f").mean().alias("true_F05")]
    return e.group_by("country").agg(out).sort("country")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--single", action="store_true",
                    help="use the singleton model on test (off by default: train has no stored OOF for it)")
    a = ap.parse_args()
    w = a.work_dir
    params = json.load(open(os.path.join(w, "cv_report.json")))
    params = params[params["chosen"]]["params"]
    pl.Config.set_tbl_cols(30), pl.Config.set_tbl_width_chars(250), pl.Config.set_float_precision(4)

    for split in ("train", "test"):
        s1c = pl.read_parquet(os.path.join(w, "norm", f"{split}_s1.parquet"), columns=["country"]) \
            .with_row_index("i").with_columns(pl.col("i").cast(pl.UInt32))
        if split == "train":
            d = pl.read_parquet(os.path.join(w, "train_oof.parquet"), columns=["i", "j", "y", "p2", "is_exp"]) \
                .rename({"p2": "p"})
            hm = None  # the train OOF singleton scores are not kept on disk
        else:
            d = pl.read_parquet(os.path.join(w, "test_scores.parquet"), columns=["i", "j", "p", "seg_exp"]) \
                .rename({"seg_exp": "is_exp"})
            hp = os.path.join(w, "test_has_match.parquet")
            hm = pl.read_parquet(hp) if (os.path.exists(hp) and a.single) else None
        # decision without segment scales (not all flags are stored for train) -> comparable on both
        prm = {k: v for k, v in params.items() if not k.startswith("seg_")}
        pred = decide(d.select("i", "j"), d["p"].to_numpy(), prm, None, hm)
        print(f"\n=== {split}  (decision {prm}, singleton model {'on' if hm is not None else 'off'})")
        nt = None
        if split == "train":
            ids = lambda f: pl.read_parquet(os.path.join(w, "norm", f), columns=["id"])  # noqa: E731
            truth = load_truth(a.data_dir, ids("train_s1.parquet"), ids("train_tg.parquet"))
            nt = truth.group_by("i").agg(pl.len().alias("n_true"))
        print(profile(d.select([c for c in ("i", "j", "p", "y") if c in d.columns]), pred, s1c, nt))
        if split == "train":
            print("check macro-F0.5 (all train S1, no seg scales):",
                  round(macro_f05(pred, truth, s1c["i"]), 4))
        del d, pred


if __name__ == "__main__":
    main()
