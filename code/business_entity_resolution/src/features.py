"""Vectorised pair features (rapidfuzz.cpdist + polars) and group-context features.

Scales to tens of millions of pairs: string scorers run multithreaded in C, set/number
features are polars list expressions, and the caller processes pairs in chunks.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import KEY_CAPS, _skeleton

REC_COLS = ["country", "name_n", "name_core", "name_ph", "name_ns", "name_acr", "name_vars",
            "addr_core", "addr_full", "has_landmark", "postal", "first_num", "nums", "id"]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _alt(vars_col: pl.Expr) -> pl.Expr:
    """Last name variant (DBA / trade name part when present, else the full name)."""
    return vars_col.str.split(" | ").list.last()


def token_idf(s1: pl.DataFrame, tg: pl.DataFrame, col: str, token_re: str) -> pl.DataFrame:
    """(tok, w) inverse document frequency of the tokens of `col` over all records of the split."""
    n = s1.height + tg.height
    t = (pl.concat([s1.select(col), tg.select(col)]).with_row_index("r")
         .select("r", tok=pl.col(col).str.split(" ")).explode("tok")
         .filter(pl.col("tok").str.contains(token_re)).unique())
    return t.group_by("tok").agg(pl.len().alias("df")).select("tok", w=(n / pl.col("df")).log().cast(pl.Float32))


def _idf_overlap(l1: pl.Series, l2: pl.Series, idf: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Rarity-weighted overlap of two aligned token-list columns.

    Sharing a rare token ("zephay") is far stronger evidence than sharing a common one
    ("traders"); an unshared rare token is strong evidence against a match."""
    n = len(l1)

    def expl(col):
        # inner join: only tokens of the idf vocabulary (e.g. alphabetic address words) count
        return (pl.DataFrame({"r": np.arange(n, dtype=np.uint32), "tok": col}).explode("tok").drop_nulls()
                .unique().join(idf, on="tok", how="inner"))

    t1, t2 = expl(l1), expl(l2)
    inter = t1.join(t2.select("r", "tok"), on=["r", "tok"], how="semi")
    u1 = t1.join(t2.select("r", "tok"), on=["r", "tok"], how="anti")
    u2 = t2.join(t1.select("r", "tok"), on=["r", "tok"], how="anti")
    base = pl.DataFrame({"r": np.arange(n, dtype=np.uint32)})
    agg = lambda d, a, b: d.group_by("r").agg(pl.col("w").sum().alias(a), pl.col("w").max().alias(b))  # noqa: E731
    d = (base.join(agg(t1, "w1", "_m1"), on="r", how="left").join(agg(t2, "w2", "_m2"), on="r", how="left")
         .join(agg(inter, "wi", "wimax"), on="r", how="left").join(agg(u1, "_s1", "um1"), on="r", how="left")
         .join(agg(u2, "_s2", "um2"), on="r", how="left").sort("r")
         .with_columns(pl.col("wi", "wimax", "um1", "um2").fill_null(0.0)))
    return d.select(
        (pl.col("wi") / (pl.col("w1") + pl.col("w2") - pl.col("wi"))).alias(f"{prefix}_wjacc"),
        (pl.col("wi") / pl.min_horizontal("w1", "w2")).alias(f"{prefix}_wcontain"),
        pl.col("wimax").alias(f"{prefix}_shared_max_idf"),
        pl.max_horizontal("um1", "um2").alias(f"{prefix}_unshared_max_idf"),
    )


def pair_features(s1: pl.DataFrame, tg: pl.DataFrame, cand: pl.DataFrame,
                  name_freq1: pl.DataFrame, name_freqt: pl.DataFrame,
                  name_idf: pl.DataFrame, addr_idf: pl.DataFrame) -> pl.DataFrame:
    """cand: (i, j, nkeys, keymask, bscore). Returns float32 feature frame aligned with cand."""
    A = s1.select(REC_COLS)[cand["i"]]
    B = tg.select(REC_COLS)[cand["j"]]
    P = pl.concat([A, B.rename({c: c + "_t" for c in REC_COLS})], how="horizontal")

    a, b = P["name_core"].to_list(), P["name_core_t"].to_list()
    an, bn = P["name_n"].to_list(), P["name_n_t"].to_list()
    ap, bp = P["name_ph"].to_list(), P["name_ph_t"].to_list()
    ans, bns = P["name_ns"].to_list(), P["name_ns_t"].to_list()
    aa, ba = P["addr_core"].to_list(), P["addr_core_t"].to_list()
    af, bf = P["addr_full"].to_list(), P["addr_full_t"].to_list()
    alt = P.select(_alt(pl.col("name_vars")).alias("x"), _alt(pl.col("name_vars_t")).alias("y"))
    ax, bx = alt["x"].to_list(), alt["y"].to_list()

    F = {
        "n_ratio": _cp(a, b, fuzz.ratio),
        "n_partial": _cp(a, b, fuzz.partial_ratio),
        "n_tsort": _cp(a, b, fuzz.token_sort_ratio),
        "n_tset": _cp(a, b, fuzz.token_set_ratio),
        "n_jw": _cp(a, b, JaroWinkler.normalized_similarity),
        "n_lev_norm": _cp(a, b, Levenshtein.normalized_distance),
        "n_full_ratio": _cp(an, bn, fuzz.ratio),
        "n_ph_ratio": _cp(ap, bp, fuzz.ratio),
        "n_ph_tsort": _cp(ap, bp, fuzz.token_sort_ratio),
        "n_nospace_ratio": _cp(ans, bns, fuzz.ratio),
        "n_nospace_partial": _cp(ans, bns, fuzz.partial_ratio),
        "n_var_best": np.maximum(_cp(ax, bn, fuzz.token_sort_ratio), _cp(an, bx, fuzz.token_sort_ratio)),
        "a_ratio": _cp(aa, ba, fuzz.ratio),
        "a_tset": _cp(aa, ba, fuzz.token_set_ratio),
        "a_tsort": _cp(aa, ba, fuzz.token_sort_ratio),
        "a_partial": _cp(aa, ba, fuzz.partial_ratio),
        "a_lev_norm": _cp(aa, ba, Levenshtein.normalized_distance),
        "a_full_tset": _cp(af, bf, fuzz.token_set_ratio),
    }
    # consonant skeletons: lets transliterated native-script names be compared with English ones
    sk = P.select(x=_skeleton(pl.col("name_core")), y=_skeleton(pl.col("name_core_t")))
    skx, sky = sk["x"].to_list(), sk["y"].to_list()
    F["n_skel_ratio"] = _cp(skx, sky, fuzz.ratio)
    F["n_skel_tset"] = _cp(skx, sky, fuzz.token_set_ratio)
    # house numbers: how far apart (hard negatives are near-copies with a shifted number)
    fa, fb = P["first_num"].to_list(), P["first_num_t"].to_list()
    F["a_num_lev"] = _cp(fa, fb, Levenshtein.distance)
    del skx, sky, fa, fb
    del a, b, an, bn, ap, bp, ans, bns, aa, ba, af, bf, ax, bx

    tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))  # noqa: E731
    atok = lambda c: tok(c).list.eval(pl.element().filter(pl.element().str.contains("^[a-z]{2,}$")))  # noqa: E731
    X = P.lazy().with_columns(
        nt1=tok("name_core"), nt2=tok("name_core_t"), at1=atok("addr_core"), at2=atok("addr_core_t"),
        nu1=tok("nums"), nu2=tok("nums_t"),
        lg1=tok("name_n").list.set_difference(tok("name_core")),
        lg2=tok("name_n_t").list.set_difference(tok("name_core_t")),
    ).with_columns(
        n_inter=pl.col("nt1").list.set_intersection("nt2").list.len(),
        n_union=pl.col("nt1").list.set_union("nt2").list.len(),
        n_min=pl.min_horizontal(pl.col("nt1").list.len(), pl.col("nt2").list.len()),
        a_inter=pl.col("at1").list.set_intersection("at2").list.len(),
        a_union=pl.col("at1").list.set_union("at2").list.len(),
        a_min=pl.min_horizontal(pl.col("at1").list.len(), pl.col("at2").list.len()),
        u_inter=pl.col("nu1").list.set_intersection("nu2").list.len(),
        u_union=pl.col("nu1").list.set_union("nu2").list.len(),
    ).select(
        n_core_eq=(pl.col("name_core") == pl.col("name_core_t")),
        n_full_eq=(pl.col("name_n") == pl.col("name_n_t")),
        n_ph_eq=(pl.col("name_ph") == pl.col("name_ph_t")),
        n_ns_eq=(pl.col("name_ns") == pl.col("name_ns_t")),
        n_jacc=pl.col("n_inter") / pl.col("n_union"),
        n_contain=pl.col("n_inter") / pl.col("n_min"),
        n_inter=pl.col("n_inter"),
        n_first_eq=pl.col("nt1").list.first() == pl.col("nt2").list.first(),
        n_acr=((pl.col("name_acr") != "") & (pl.col("name_acr") == pl.col("name_ns_t")))
        | ((pl.col("name_acr_t") != "") & (pl.col("name_acr_t") == pl.col("name_ns"))),
        n_len1=pl.col("name_core").str.len_chars(),
        n_len2=pl.col("name_core_t").str.len_chars(),
        n_ntok_diff=(pl.col("nt1").list.len() - pl.col("nt2").list.len()).abs(),
        n_legal_conflict=(pl.col("lg1").list.len() > 0) & (pl.col("lg2").list.len() > 0)
        & (pl.col("lg1").list.set_intersection("lg2").list.len() == 0),
        n_nonascii_t=pl.col("name_vars_t").str.contains("[^a-z0-9 |]"),
        a_jacc=pl.col("a_inter") / pl.col("a_union"),
        a_contain=pl.col("a_inter") / pl.col("a_min"),
        a_inter=pl.col("a_inter"),
        a_empty1=pl.col("addr_core").str.len_chars() < 3,
        a_empty2=pl.col("addr_core_t").str.len_chars() < 3,
        a_len1=pl.col("addr_core").str.len_chars(),
        a_len2=pl.col("addr_core_t").str.len_chars(),
        a_postal_eq=pl.when((pl.col("postal") != "") & (pl.col("postal_t") != ""))
        .then(pl.col("postal") == pl.col("postal_t")),
        a_postal_missing=(pl.col("postal") == "").cast(pl.Int8) + (pl.col("postal_t") == "").cast(pl.Int8),
        a_num_eq=pl.when((pl.col("first_num") != "") & (pl.col("first_num_t") != ""))
        .then(pl.col("first_num") == pl.col("first_num_t")),
        a_num_in=pl.when(pl.col("first_num_t") != "").then(pl.col("nu1").list.contains(pl.col("first_num_t"))),
        a_num_absdiff=(pl.col("first_num").cast(pl.Float64, strict=False)
                       - pl.col("first_num_t").cast(pl.Float64, strict=False)).abs().log1p(),
        a_num_reldiff=(pl.col("first_num").cast(pl.Float64, strict=False)
                       - pl.col("first_num_t").cast(pl.Float64, strict=False)).abs()
        / pl.max_horizontal(pl.col("first_num").cast(pl.Float64, strict=False),
                            pl.col("first_num_t").cast(pl.Float64, strict=False), pl.lit(1.0)),
        a_nums_jacc=pl.when(pl.col("u_union") > 0).then(pl.col("u_inter") / pl.col("u_union")),
        a_nums_disjoint=pl.when((pl.col("nu1").list.len() > 0) & (pl.col("nu2").list.len() > 0))
        .then(pl.col("u_inter") == 0),
        a_landmark=pl.col("has_landmark").cast(pl.Int8) + pl.col("has_landmark_t").cast(pl.Int8),
        # last address token is the state / region code in all countries after normalisation
        a_last_eq=pl.when((pl.col("at1").list.len() > 0) & (pl.col("at2").list.len() > 0))
        .then(pl.col("at1").list.last() == pl.col("at2").list.last()),
        a_tail_overlap=pl.col("at1").list.tail(3).list.set_intersection(pl.col("at2").list.tail(3)).list.len(),
        same_country=pl.col("country") == pl.col("country_t"),
        tgt_is_s3=pl.col("id_t").str.starts_with("S3"),
    ).collect()

    out = pl.DataFrame(F).with_columns(
        na_ratio=pl.col("n_ratio") * 0.6 + pl.col("a_ratio") * 0.4)
    tl = lambda c: P[c].str.split(" ")  # noqa: E731
    W = pl.concat([_idf_overlap(tl("name_core"), tl("name_core_t"), name_idf, "nw"),
                   _idf_overlap(tl("addr_core"), tl("addr_core_t"), addr_idf, "aw")], how="horizontal")
    out = pl.concat([out, X, W, cand.select("nkeys", "bscore")], how="horizontal")
    # explicit name x address interactions (a strong name match means little with a different address)
    out = out.with_columns(
        x_tset=pl.col("n_tset") * pl.col("a_tset") / 100.0,
        x_wjacc=pl.col("nw_wjacc") * pl.col("aw_wjacc"),
        x_skel_addr=pl.col("n_skel_tset") * pl.col("a_tset") / 100.0,
    )
    # name frequency: chains / generic names need address evidence
    nf = (cand.select("i", "j")
          .join(name_freq1, left_on="i", right_on="idx", how="left").rename({"freq": "freq_name_s1"})
          .join(name_freqt, left_on="j", right_on="idx", how="left").rename({"freq": "freq_name_tg"}))
    out = out.with_columns(nf["freq_name_s1"], nf["freq_name_tg"])
    out = out.with_columns(freq_name_min=pl.min_horizontal("freq_name_s1", "freq_name_tg"))
    km = cand["keymask"].to_numpy()
    out = out.with_columns([pl.Series(f"key{k}", ((km >> k) & 1).astype(np.int8))  # which blocking keys agreed
                            for k in range(len(KEY_CAPS))])
    return out.cast({c: pl.Float32 for c in out.columns})


def name_frequencies(s1: pl.DataFrame, tg: pl.DataFrame):
    """How often each core name occurs (per country) among S1 and among targets."""
    both = pl.concat([s1.select("country", "name_core"), tg.select("country", "name_core")])
    f1 = s1.group_by("country", "name_core").agg(pl.len().alias("freq"))
    ft = both.group_by("country", "name_core").agg(pl.len().alias("freq"))
    a = s1.select("country", "name_core").with_row_index("idx").join(f1, on=["country", "name_core"], how="left")
    b = tg.select("country", "name_core").with_row_index("idx").join(ft, on=["country", "name_core"], how="left")
    return a.select("idx", "freq").sort("idx"), b.select("idx", "freq").sort("idx")


GROUP_BASE = ["n_tsort", "a_tset", "bscore", "n_var_best"]


def group_features(slim: pl.DataFrame) -> pl.DataFrame:
    """Rank / gap of a pair's similarity within its S1 group and within its target group.

    slim: (i, j, *GROUP_BASE) for ALL candidate pairs of the split (row-aligned)."""
    exprs = [pl.len().over("i").alias("cands_per_s1"), pl.len().over("j").alias("cands_per_tg")]
    for c in GROUP_BASE:
        exprs += [
            pl.col(c).rank("min", descending=True).over("i").alias(f"rank1_{c}"),
            pl.col(c).rank("min", descending=True).over("j").alias(f"rankt_{c}"),
            (pl.col(c) - pl.col(c).max().over("i")).alias(f"gap1_{c}"),
            (pl.col(c) - pl.col(c).max().over("j")).alias(f"gapt_{c}"),
        ]
    out = slim.select(exprs)
    return out.cast({c: pl.Float32 for c in out.columns})


def context_features(ij: pl.DataFrame, p: np.ndarray, tg: pl.DataFrame, top=3, min_p=0.03,
                     chunk=5_000_000) -> pl.DataFrame:
    """Stage-2 features from stage-1 probabilities (out-of-fold on train). ij: (i, j) for all pairs.

    Includes *coherence*: how similar this target is to the S1 entity's other confident targets
    (true matches of one business tend to resemble each other; a stray chain branch does not)."""
    g = ij.select("i", "j").with_columns(p=pl.Series(p, dtype=pl.Float32))

    def top2(by):
        s = pl.col("p").sort(descending=True)
        return [s.first().over(by).alias("t1"), s.get(1, null_on_oob=True).over(by).fill_null(0).alias("t2")]

    gi = g.with_columns(top2("i")).select(
        p_margin_s1=pl.col("p") - pl.when(pl.col("p") >= pl.col("t1")).then(pl.col("t2")).otherwise(pl.col("t1")))
    gj = g.with_columns(top2("j")).select(
        p_margin_tg=pl.col("p") - pl.when(pl.col("p") >= pl.col("t1")).then(pl.col("t2")).otherwise(pl.col("t1")))
    C = g.select(
        p1="p",
        p_rank_s1=pl.col("p").rank("min", descending=True).over("i"),
        p_rank_tg=pl.col("p").rank("min", descending=True).over("j"),
        p_max_s1=pl.col("p").max().over("i"),
        p_max_tg=pl.col("p").max().over("j"),
        p_sum_s1=pl.col("p").sum().over("i"),
        p_sum_tg=pl.col("p").sum().over("j"),
        p_n50_s1=(pl.col("p") > 0.5).sum().over("i"),
        p_n50_tg=(pl.col("p") > 0.5).sum().over("j"),
    )

    # coherence (only for pairs with non-negligible p; others stay null)
    g = g.with_row_index("r")
    T = g.sort("p", descending=True).group_by("i").head(top).select("i", u="j", pu="p")
    rows = g.filter(pl.col("p") >= min_p).select("r", "i", "j")
    names, addrs = tg["name_core"], tg["addr_core"]
    res = []
    for s in range(0, rows.height, chunk):
        q = rows.slice(s, chunk).join(T, on="i").filter(pl.col("j") != pl.col("u"))
        if q.height == 0:
            continue
        jn, un = names.gather(q["j"]).to_list(), names.gather(q["u"]).to_list()
        sn = process.cpdist(jn, un, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
        del jn, un
        ja, ua = addrs.gather(q["j"]).to_list(), addrs.gather(q["u"]).to_list()
        sa = process.cpdist(ja, ua, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
        del ja, ua
        pu = q["pu"].to_numpy()
        res.append(q.select("r").with_columns(
            coh_n=pl.Series(sn * pu), coh_a=pl.Series(sa * pu),
            coh_sim_n=pl.Series(sn), coh_sim_a=pl.Series(sa))
            .group_by("r").agg(pl.all().max()))
    coh = pl.concat(res) if res else pl.DataFrame(schema={"r": pl.UInt32, "coh_n": pl.Float32, "coh_a": pl.Float32,
                                                         "coh_sim_n": pl.Float32, "coh_sim_a": pl.Float32})
    coh = g.select("r").join(coh, on="r", how="left", maintain_order="left").drop("r")
    out = pl.concat([C, gi, gj, coh], how="horizontal")
    return out.cast({c: pl.Float32 for c in out.columns})
