"""Country-specific decision adjustment on top of a blended submission (for an unseen country).

France has no training labels, so its decision threshold is inherited from US/India. This script
re-applies the chosen blend with the probabilities of one country's pairs scaled by a factor
(< 1: fewer, more certain links; > 1: more links), leaving every other country's rows unchanged.

    python src/country_adjust.py --works work work_v4 work_v5 --candidate 'work+work_v5' \
        --country france --factors 0.8 1.25 --out-prefix output_fr

--agree-with WORK: additionally keep a link of that country only if the same decision made with WORK's
own scores for that country (e.g. the run trained after the country's normalisation fix) keeps it too.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend_oof import _ids, scores, union  # noqa: E402
from pipeline import decide, singleton_model, write_lists  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--works", nargs="+", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--country", default="france")
    ap.add_argument("--factors", nargs="+", type=float, required=True)
    ap.add_argument("--out-prefix", default=None, help="write <prefix>_<factor>/ for each factor")
    ap.add_argument("--agree-with", default=None, help="run folder whose own scores must also select a link")
    a = ap.parse_args()
    ref = a.works[-1]
    c = joblib.load(os.path.join(ref, "blend_eval.joblib"))["report"][a.candidate]
    u = union([scores(w, "test", ref) for w in c["works"]], c["weights"])
    ij, p0 = u.select("i", "j"), u["p"].to_numpy().astype(np.float32)
    s1t = pl.read_parquet(os.path.join(ref, "norm", "test_s1.parquet"), columns=["id", "name_core", "country"])
    is_c = (s1t["country"] == a.country).to_numpy()[ij["i"].to_numpy()]
    p_alt = None
    if a.agree_with:  # the country's pairs scored by one run only (others keep the blend)
        alt = scores(a.agree_with, "test", ref).rename({"p": "pw"})
        pw = ij.join(alt, on=["i", "j"], how="left")["pw"].to_numpy()
        p_alt = np.where(is_c & ~np.isnan(pw), pw, p0).astype(np.float32)
    tg = _ids(ref, "test_tg.parquet").to_list()
    base = None
    for f in [1.0] + a.factors:
        p = p0.copy()
        p[is_c] = np.clip(p[is_c] * f, 0, 1)
        hm = singleton_model(ij, p, None, s1t, models=c["sm"])[0]
        pred = decide(ij, p, c["params"], None, hm)
        if p_alt is not None and f != 1.0:
            q = p_alt.copy()
            q[is_c] = np.clip(q[is_c] * f, 0, 1)
            pa = decide(ij, q, c["params"], None, singleton_model(ij, q, None, s1t, models=c["sm"])[0])
            cmask = pl.DataFrame({"i": np.where((s1t["country"] == a.country).to_numpy())[0].astype(np.uint32)})
            pred = pl.concat([pred.join(cmask, on="i", how="anti"),
                              pred.join(cmask, on="i", how="semi").join(pa, on=["i", "j"], how="semi")]).sort("i", "j")
        ci = pl.DataFrame({"i": np.where((s1t["country"] == a.country).to_numpy())[0].astype(np.uint32)})
        pc = pred.join(ci, on="i", how="semi")
        empty = 1 - pc["i"].n_unique() / ci.height
        if base is None:
            base = pred
            print(f"factor 1.00 (current): {a.country} links={pc.height} empty={empty:.4f}")
            continue
        others_same = pred.join(ci, on="i", how="anti").sort("i", "j").equals(base.join(ci, on="i", how="anti").sort("i", "j"))
        added = pc.join(base, on=["i", "j"], how="anti").height
        removed = base.join(ci, on="i", how="semi").join(pc, on=["i", "j"], how="anti").height
        print(f"factor {f:.2f}: {a.country} links={pc.height} (+{added} / -{removed}) empty={empty:.4f}  other countries unchanged: {others_same}")
        if a.out_prefix:
            out = f"{a.out_prefix}_{f:.2f}"
            os.makedirs(out, exist_ok=True)
            write_lists(os.path.join(out, "matching_results.tsv"), s1t["id"].to_list(), pred, tg, "matched_entity_ids")
            write_lists(os.path.join(out, "candidate_pairs.tsv"), s1t["id"].to_list(), ij, tg, "candidate_entity_ids")
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
