"""Pairwise + group-context features for candidate pairs."""
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import Vectors


def _jacc(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def _num_feats(n1, n2):
    s1, s2 = set(n1), set(n2)
    if not s1 or not s2:
        return np.nan, np.nan
    inter = len(s1 & s2)
    return inter / len(s1 | s2), float(inter == 0)


def pair_features(s1: pd.DataFrame, tg: pd.DataFrame, cand: pd.DataFrame, vec: Vectors) -> pd.DataFrame:
    i, j = cand.i.values, cand.j.values
    F = pd.DataFrame(index=cand.index)
    for key in Vectors.SPACES:
        F["cos_" + key] = vec.pair_cos(key, i, j)

    A, B = s1.iloc[i].reset_index(drop=True), tg.iloc[j].reset_index(drop=True)
    nc1, nc2 = A.name_core.values, B.name_core.values
    nn1, nn2 = A.name_n.values, B.name_n.values
    ph1, ph2 = A.name_ph.values, B.name_ph.values
    ac1, ac2 = A.addr_core.values, B.addr_core.values
    af1, af2 = A.addr_full.values, B.addr_full.values
    v1, v2 = A.name_vars.values, B.name_vars.values

    F["n_ratio"] = [fuzz.ratio(a, b) for a, b in zip(nc1, nc2)]
    F["n_partial"] = [fuzz.partial_ratio(a, b) for a, b in zip(nc1, nc2)]
    F["n_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(nc1, nc2)]
    F["n_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(nc1, nc2)]
    F["n_jw"] = [JaroWinkler.similarity(a, b) for a, b in zip(nc1, nc2)]
    F["n_full_ratio"] = [fuzz.ratio(a, b) for a, b in zip(nn1, nn2)]
    F["n_ph_ratio"] = [fuzz.ratio(a, b) for a, b in zip(ph1, ph2)]
    F["n_ph_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(ph1, ph2)]
    F["n_nospace_ratio"] = [fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")) for a, b in zip(nc1, nc2)]
    F["n_lev"] = [Levenshtein.distance(a, b) for a, b in zip(nc1, nc2)]
    F["n_var_best"] = [max(fuzz.token_sort_ratio(x, y) for x in a for y in b) for a, b in zip(v1, v2)]
    F["n_core_eq"] = (nc1 == nc2).astype(np.float32)
    F["n_full_eq"] = (nn1 == nn2).astype(np.float32)
    t1 = [set(x.split()) for x in nc1]
    t2 = [set(x.split()) for x in nc2]
    F["n_jacc"] = [_jacc(a, b) for a, b in zip(t1, t2)]
    F["n_contain"] = [len(a & b) / max(min(len(a), len(b)), 1) for a, b in zip(t1, t2)]
    F["n_first_eq"] = [float(bool(a) and bool(b) and a.split()[0] == b.split()[0]) for a, b in zip(nc1, nc2)]
    F["n_first_ratio"] = [fuzz.ratio(a.split()[0], b.split()[0]) if a and b else np.nan for a, b in zip(nc1, nc2)]
    F["n_acr"] = [float(bool(x and x == b.replace(" ", "")) or bool(y and y == a.replace(" ", "")))
                  for a, b, x, y in zip(nc1, nc2, A.name_acr.values, B.name_acr.values)]
    F["n_len1"] = [len(x) for x in nc1]
    F["n_len2"] = [len(x) for x in nc2]
    F["n_ntok_diff"] = [abs(len(a) - len(b)) for a, b in zip(t1, t2)]
    lg1 = [set(x.split()) - set(c.split()) for x, c in zip(nn1, nc1)]
    lg2 = [set(x.split()) - set(c.split()) for x, c in zip(nn2, nc2)]
    F["n_legal_conflict"] = [float(bool(a) and bool(b) and not (a & b)) for a, b in zip(lg1, lg2)]

    F["a_ratio"] = [fuzz.ratio(a, b) for a, b in zip(ac1, ac2)]
    F["a_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(ac1, ac2)]
    F["a_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(ac1, ac2)]
    F["a_partial"] = [fuzz.partial_ratio(a, b) if a and b else np.nan for a, b in zip(ac1, ac2)]
    F["a_full_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(af1, af2)]
    p1, p2 = A.addr_postal.values, B.addr_postal.values
    both = (p1 != "") & (p2 != "")
    F["a_postal_eq"] = np.where(both, (p1 == p2).astype(np.float32), np.nan)
    F["a_postal_pref3"] = np.where(both, [float(a[:3] == b[:3]) for a, b in zip(p1, p2)], np.nan)
    F["a_postal_missing"] = ((p1 == "").astype(int) + (p2 == "").astype(int)).astype(np.float32)
    fn1, fn2 = A.addr_first_num.values, B.addr_first_num.values
    bothn = (fn1 != "") & (fn2 != "")
    F["a_num_eq"] = np.where(bothn, (fn1 == fn2).astype(np.float32), np.nan)
    nf = [_num_feats(a, b) for a, b in zip(A.addr_nums.values, B.addr_nums.values)]
    F["a_nums_jacc"] = [x[0] for x in nf]
    F["a_nums_disjoint"] = [x[1] for x in nf]
    F["a_landmark"] = ((A.addr_landmark.values != "").astype(int) + (B.addr_landmark.values != "").astype(int))
    F["a_len1"] = [len(x) for x in ac1]
    F["a_len2"] = [len(x) for x in ac2]
    at1 = [set(x.split()) for x in ac1]
    at2 = [set(x.split()) for x in ac2]
    F["a_contain"] = [len(a & b) / max(min(len(a), len(b)), 1) for a, b in zip(at1, at2)]

    F["same_country"] = (A.country.values == B.country.values).astype(np.float32)
    F["tgt_is_s3"] = (B.src.values == "S3").astype(np.float32)

    # name frequency => chains / generic names need address evidence
    cnt1 = Counter(s1.name_core)
    cntt = Counter(tg.name_core)
    F["freq_name_s1"] = [cnt1[x] for x in nc1]
    F["freq_name_tg"] = [cntt[x] for x in nc2]

    # candidate-set context
    g = pd.DataFrame({"i": i, "j": j, "n": F.cos_name_char.values, "a": F.cos_addr_char.values,
                      "w": F.cos_all_word.values})
    F["cands_per_s1"] = g.groupby("i")["j"].transform("size").values
    F["cands_per_tg"] = g.groupby("j")["i"].transform("size").values
    for col in ("n", "a", "w"):
        F[f"rank1_{col}"] = g.groupby("i")[col].rank(ascending=False, method="min").values
        F[f"rankt_{col}"] = g.groupby("j")[col].rank(ascending=False, method="min").values
        F[f"gap1_{col}"] = (g[col] - g.groupby("i")[col].transform("max")).values
        F[f"gapt_{col}"] = (g[col] - g.groupby("j")[col].transform("max")).values
    return F.astype(np.float32)


def context_features(cand: pd.DataFrame, p: np.ndarray, vec: Vectors, top=3) -> pd.DataFrame:
    """Stage-2 features built from stage-1 probabilities (must be out-of-fold on train)."""
    g = pd.DataFrame({"i": cand.i.values, "j": cand.j.values, "p": p})
    C = pd.DataFrame(index=cand.index)
    C["p1"] = p
    gi, gj = g.groupby("i")["p"], g.groupby("j")["p"]
    C["p_rank_s1"] = gi.rank(ascending=False, method="min").values
    C["p_rank_tg"] = gj.rank(ascending=False, method="min").values
    C["p_max_s1"] = gi.transform("max").values
    C["p_max_tg"] = gj.transform("max").values
    C["p_sum_s1"] = gi.transform("sum").values
    C["p_sum_tg"] = gj.transform("sum").values
    C["p_n50_s1"] = g.assign(h=g.p > 0.5).groupby("i")["h"].transform("sum").values
    C["p_n50_tg"] = g.assign(h=g.p > 0.5).groupby("j")["h"].transform("sum").values
    # best competing S1 for this target (excluding self)
    s = g.sort_values("p", ascending=False)
    first = s.groupby("j").head(1).set_index("j")
    second = s.groupby("j").nth(1).set_index("j")
    top1 = first.p.reindex(g.j).values
    top2 = second.p.reindex(g.j).fillna(0).values
    other = np.where(g.i.values == first.i.reindex(g.j).values, top2, top1)
    C["p_margin_tg"] = p - other
    first_i = s.groupby("i").head(1).set_index("i")
    second_i = s.groupby("i").nth(1).set_index("i")
    t1 = first_i.p.reindex(g.i).values
    t2 = second_i.p.reindex(g.i).fillna(0).values
    C["p_margin_s1"] = p - np.where(g.j.values == first_i.j.reindex(g.i).values, t2, t1)

    # coherence: similarity of this target to the S1's other confident targets
    topk = s.groupby("i").head(top)[["i", "j", "p"]].rename(columns={"j": "u", "p": "pu"})
    pairs = g.reset_index().merge(topk, on="i")
    pairs = pairs[pairs.j != pairs.u]
    mt = vec.mt["all_word"]
    sim = np.asarray(mt[pairs.j.values].multiply(mt[pairs.u.values]).sum(1)).ravel() if len(pairs) else np.array([])
    pairs["coh"] = sim * pairs.pu.values
    pairs["sim"] = sim
    agg = pairs.groupby("index").agg(coh=("coh", "max"), coh_sim=("sim", "max"))
    C["coh"] = agg.coh.reindex(C.index).values
    C["coh_sim"] = agg.coh_sim.reindex(C.index).values
    return C.astype(np.float32)
