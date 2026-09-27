"""Entity-mix shift: describe each S1 entity by model-free counts over its candidates, then
re-weight train entities so their mix matches test (per country, histogram density ratio).

    python src/entity_mix.py --work-dir work            # writes <work>/entity_desc_{train,test}.parquet
                                                        # and <work>/entity_weights.parquet (train S1)

A weighted out-of-fold macro-F0.5 then estimates the test score *if* the train->test shift is a
change in the mix of entities (more duplicates, more decoys per business) rather than new noise.
"""
import argparse
import glob
import os

import polars as pl

DESC = ["n_near", "n_decoy", "n_cand_b"]


def describe(work: str, split: str) -> pl.DataFrame:
    """Per S1: # near-exact candidates, # decoy-like candidates (same name+address, other number),
    candidate-count bucket. Uses the cached base candidates + pair features."""
    cand = pl.read_parquet(os.path.join(work, f"{split}_cand.parquet"), columns=["i"])
    parts = sorted(glob.glob(os.path.join(work, f"{split}_feat_v*", "part_*.parquet")))
    F = pl.concat([pl.read_parquet(p, columns=["n_tsort", "a_tset", "a_num_eq"]) for p in parts])
    d = pl.concat([cand, F], how="horizontal_extend")
    near = (pl.col("n_tsort") >= 90) & (pl.col("a_tset") >= 90)
    decoy = (pl.col("n_tsort") >= 85) & (pl.col("a_tset") >= 80) & (pl.col("a_num_eq") == 0)
    e = d.group_by("i").agg(n_near=near.sum(), n_decoy=decoy.sum(), n_cand=pl.len())
    s1 = pl.read_parquet(os.path.join(work, "norm", f"{split}_s1.parquet"), columns=["country"]) \
        .with_row_index("i").with_columns(pl.col("i").cast(pl.UInt32))
    e = s1.join(e, on="i", how="left").fill_null(0)
    return e.with_columns(
        n_near=pl.col("n_near").clip(0, 8), n_decoy=pl.col("n_decoy").clip(0, 4),
        n_cand_b=pl.col("n_cand").cut([10, 20, 29], labels=["<=10", "11-20", "21-29", "30"]).cast(pl.String))


def weights(tr: pl.DataFrame, te: pl.DataFrame, country_map: dict, alpha=1.0) -> pl.DataFrame:
    """w(bin) = P_test(bin) / P_train(bin) per country, with additive smoothing.
    country_map: test country -> train country whose entities stand in for it."""
    out = []
    for c_te, c_tr in country_map.items():
        a = tr.filter(pl.col("country") == c_tr).group_by(DESC).len("n_tr")
        b = te.filter(pl.col("country") == c_te).group_by(DESC).len("n_te")
        h = a.join(b, on=DESC, how="full", coalesce=True).fill_null(0)
        h = h.with_columns(w=((pl.col("n_te") + alpha) / (pl.col("n_te").sum() + alpha * h.height))
                           / ((pl.col("n_tr") + alpha) / (pl.col("n_tr").sum() + alpha * h.height)))
        w = tr.filter(pl.col("country") == c_tr).join(h.select(*DESC, "w"), on=DESC, how="left")
        out.append(w.select("i", pl.lit(c_te).alias("as_country"), "w"))
    return pl.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    a = ap.parse_args()
    w = a.work_dir
    desc = {}
    for split in ("train", "test"):
        path = os.path.join(w, f"entity_desc_{split}.parquet")
        if not os.path.exists(path):
            describe(w, split).write_parquet(path)
        desc[split] = pl.read_parquet(path)
        d = desc[split]
        print(split, d.group_by("country").agg(n=pl.len(), near=pl.col("n_near").mean(),
                                               decoy=pl.col("n_decoy").mean(), cand=pl.col("n_cand").mean(),
                                               no_near=(pl.col("n_near") == 0).mean()).sort("country"))
    wt = weights(desc["train"], desc["test"], {"us": "us", "india": "india"})
    wt.write_parquet(os.path.join(w, "entity_weights.parquet"))
    print(wt.group_by("as_country").agg(w_mean=pl.col("w").mean(), w_max=pl.col("w").max(),
                                        ess=pl.col("w").sum() ** 2 / (pl.col("w") ** 2).sum()))


if __name__ == "__main__":
    main()
