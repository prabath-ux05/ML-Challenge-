"""Cluster expansion: recover matches that blocking missed, via an entity's confident matches.

A Source 1 entity's Source 2/3 records often resemble *each other* more than they resemble the
Source 1 record (e.g. a Hindi-script name whose address equals an English-named sibling's).
Given stage-1 scores, for each confident pair (i, u) we look for targets t that share a blocking
key with u and are near-duplicates of u (name or address similarity), and add (i, t) as new
candidates together with "sibling" evidence features.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from blocking import add_keys

# (key on the confident sibling u, key on the new target t)
EXP_KEYS = [
    ("core", "core"), ("ph", "ph"), ("core", "core_noaddr"), ("ph", "ph_noaddr"),
    ("core_num", "core_num"), ("ph_num", "ph_num"), ("skel", "skel"), ("skel_num", "skel_num"),
    ("skel_a", "skel_a"), ("core_a", "core_a"), ("ns7_a", "ns7_a"), ("ns7_num", "ns7_num"),
    ("na", "na"), ("nnum", "nnum"), ("anum", "anum"), ("aa", "aa"),
]
DEFAULT_EXPANSION = dict(tau=0.5, cap_t=30, min_sim=85.0, per_s1=10)


def _long(k: pl.DataFrame, idx: str, key: str) -> pl.DataFrame:
    return k.select(idx, key).explode(key).drop_nulls().rename({key: "kv"}).unique()


def _sibling_pairs(kt: pl.DataFrame, us: pl.DataFrame, cap_t: int) -> pl.DataFrame:
    """(lu, j): targets j sharing a small block (<= cap_t targets) with a confident sibling lu."""
    kq = kt.join(us, left_on="j", right_on="lu", how="semi")
    out = None
    for qk, tk in EXP_KEYS:
        T = _long(kt, "j", tk)
        T = T.join(T.group_by("kv").agg(pl.len().alias("nt")).filter(pl.col("nt") <= cap_t).select("kv"), on="kv")
        U = _long(kq, "j", qk).rename({"j": "lu"})
        p = U.join(T, on="kv").filter(pl.col("lu") != pl.col("j")).select("lu", "j").unique()
        out = p if out is None else pl.concat([out, p]).unique()
        del T, U, p
    return out


def expansion_candidates(s1: pl.DataFrame, tg: pl.DataFrame, cand: pl.DataFrame, p: np.ndarray,
                         cfg=None, log=print) -> pl.DataFrame:
    """Returns (i, j, sib_sn, sib_sa, sib_p, sib_n) for new pairs not already in `cand`."""
    cfg = {**DEFAULT_EXPANSION, **(cfg or {})}
    conf = (cand.select("i", "j").with_columns(p=pl.Series(p, dtype=pl.Float32))
            .filter(pl.col("p") >= cfg["tau"]).rename({"j": "u"}))
    log(f"  expansion: {conf.height} confident pairs (p>={cfg['tau']})")
    names, addrs = tg["name_core"], tg["addr_core"]
    ut = []
    for c in s1["country"].unique().sort():
        i_map = s1.with_row_index("gi").filter(pl.col("country") == c)
        j_map = tg.with_row_index("gj").filter(pl.col("country") == c)
        if j_map.height == 0:
            continue
        _, kt = add_keys(i_map.drop("gi"), j_map.drop("gj"))
        g2l = pl.DataFrame({"u": j_map["gj"], "lu": np.arange(j_map.height, dtype=np.uint32)})
        us = conf.select("u").unique().join(g2l, on="u").select("lu")
        pr = _sibling_pairs(kt, us, cfg["cap_t"])
        del kt
        pr = pr.select(u=j_map["gj"].gather(pr["lu"]), t=j_map["gj"].gather(pr["j"]))
        n0 = pr.height
        # verify: t must be a near-duplicate of the sibling u (name or address)
        kept = []
        step = 5_000_000
        for s in range(0, pr.height, step):
            q = pr.slice(s, step)
            sn = process.cpdist(names.gather(q["u"]).to_list(), names.gather(q["t"]).to_list(),
                                scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            sa = process.cpdist(addrs.gather(q["u"]).to_list(), addrs.gather(q["t"]).to_list(),
                                scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            keep = np.maximum(sn, sa) >= cfg["min_sim"]
            kept.append(q.filter(pl.Series(keep)).with_columns(sn=pl.Series(sn[keep]), sa=pl.Series(sa[keep])))
        pr = pl.concat(kept)
        ut.append(pr)
        log(f"  expansion {c}: {n0} sibling-target pairs, {pr.height} near-duplicates")
        del kept
    ut = pl.concat(ut) if ut else pl.DataFrame(schema={"u": pl.UInt32, "t": pl.UInt32, "sn": pl.Float32,
                                                        "sa": pl.Float32})
    new = (conf.join(ut, on="u")
           .group_by("i", "t").agg(sib_sn=pl.col("sn").max(), sib_sa=pl.col("sa").max(),
                                   sib_p=pl.col("p").max(), sib_n=pl.len())
           .rename({"t": "j"})
           .join(cand.select("i", "j"), on=["i", "j"], how="anti")
           .with_columns(score=pl.max_horizontal("sib_sn", "sib_sa") * pl.col("sib_p"))
           .sort(["i", "score"], descending=[False, True])
           .group_by("i", maintain_order=True).head(cfg["per_s1"])
           .drop("score"))
    return new.with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32),
                            pl.col("sib_sn", "sib_sa", "sib_p").cast(pl.Float32), pl.col("sib_n").cast(pl.Float32))
