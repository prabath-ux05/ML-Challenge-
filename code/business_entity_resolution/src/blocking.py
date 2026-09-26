"""Key-based blocking that scales to 10M+ records (hash joins in polars).

Every key is prefixed with the country label, so an unseen label (e.g. France) simply forms
its own blocks. Rare-token keys only use tokens that appear at least twice in the split
(a token seen once cannot link anything), which makes typo'd tokens fall through to other keys.
"""
import polars as pl

N_RARE = 3  # rarest tokens kept per record, per field


def _combos(xs, ys, sep):
    """All x+sep+y combinations as a list column (nulls dropped)."""
    return pl.concat_list([pl.when(x.is_not_null() & y.is_not_null()).then(x + sep + y)
                           for x in xs for y in ys]).list.drop_nulls()


def _pairs(xs):
    """All order-independent pairs within xs as a list column."""
    out = []
    for a in range(len(xs)):
        for b in range(a + 1, len(xs)):
            x, y = xs[a], xs[b]
            out.append(pl.when(x.is_not_null() & y.is_not_null()).then(
                pl.min_horizontal(x, y) + " " + pl.max_horizontal(x, y)))
    return pl.concat_list(out).list.drop_nulls()


def _skeleton(e: pl.Expr) -> pl.Expr:
    """Sorted 3-letter consonant stems of each token (vowels/h dropped after the first letter)."""
    for a, b in (("ght", "t"), ("ph", "f"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "j"), ("w", "v")):
        e = e.str.replace_all(a, b, literal=True)
    e = e.str.replace_all(r"([bcdfgjklmnpqrstvxz])h", "$1").str.replace_all(r"\B[aeiouyh]+", "")
    for ch in "bdfgjklmnprstv":
        e = e.str.replace_all(ch + ch, ch, literal=True)
    return e.str.split(" ").list.eval(pl.element().str.slice(0, 3)).list.sort().list.join(" ")


def _key_exprs():
    """key name -> (list-valued expression, max S1 per key value, max targets per key value).

    Keys are multi-valued so that e.g. a shared (name token, address token) links two records
    even when typos change which tokens are the rarest on either side (cross-rank matching).
    """
    nt = [pl.col(f"nt_{k}") for k in range(1, N_RARE + 1)]
    at = [pl.col(f"at_{k}") for k in range(1, N_RARE + 1)]
    nums = [pl.col("nums").str.split(" ").list.get(k, null_on_oob=True) for k in range(3)]
    nums = [pl.when(n != "").then(n) for n in nums]
    num = pl.when(pl.col("first_num") != "").then(pl.col("first_num"))
    core = pl.when(pl.col("name_core") != "").then(pl.col("name_core"))
    ph = pl.when(pl.col("name_ph") != "").then(pl.col("name_ph").str.split(" ").list.sort().list.join(" "))
    ns7 = pl.when(pl.col("name_ns").str.len_chars() >= 6).then(pl.col("name_ns").str.slice(0, 7))
    # asymmetric: S1 side always emits, target side only when its address is empty
    no_addr = (pl.col("addr_core").str.len_chars() < 3) | (pl.col("side") == 0)
    skel = pl.when(pl.col("name_core") != "").then(_skeleton(pl.col("name_core")))
    one = lambda e: pl.concat_list([e]).list.drop_nulls()  # noqa: E731
    return {
        "core": (one(core), 30, 300),
        "ph": (one(ph), 30, 300),
        "core_noaddr": (one(pl.when(no_addr).then(core)), 200, 200),
        "ph_noaddr": (one(pl.when(no_addr).then(ph)), 200, 200),
        "ntok_noaddr": (pl.when(no_addr).then(pl.concat_list(nt).list.drop_nulls()), 50, 200),
        "core_num": (one(core + "#" + num), 15, 150),
        "ph_num": (one(ph + "#" + num), 15, 150),
        "ns7_a": (_combos([ns7], at[:2], "@"), 15, 150),
        "ns7_num": (one(ns7 + "#" + num), 15, 150),
        "core_a": (_combos([core], at[:2], "@"), 30, 300),
        "na": (_combos(nt[:2], at, "@"), 15, 150),
        "nnum": (_combos(nt[:2], nums, "#"), 15, 150),
        "anum": (_combos(at, nums, "#"), 15, 150),
        "aa": (_pairs(at), 15, 150),
        "nn": (_pairs(nt), 15, 150),
        # consonant skeleton of the name: transliteration-robust ("brait knsaltensi" ~ "bright consultancy")
        "skel": (one(skel), 15, 150),
        "skel_num": (one(skel + "#" + num), 15, 150),
        "skel_a": (_combos([skel], at[:2], "@"), 15, 150),
    }


KEY_CAPS = {k: (c1, ct) for k, (_, c1, ct) in _key_exprs().items()}
DEFAULT_BLOCKING = dict(max_per_s1=30)


def _rare_tokens(df: pl.DataFrame, col: str, token_re: str, n: int, prefix: str) -> pl.DataFrame:
    """Per row: the n rarest tokens (document frequency >= 2 within country) -> prefix_1..n."""
    t = (df.select("rid", "country", pl.col(col).str.split(" ").alias("tok"))
         .explode("tok")
         .filter(pl.col("tok").str.contains(token_re))
         .unique(["rid", "tok"]))
    dfreq = t.group_by("country", "tok").agg(pl.len().alias("df"))
    t = (t.join(dfreq, on=["country", "tok"]).filter(pl.col("df") >= 2)
         .sort(["rid", "df", "tok"]).group_by("rid", maintain_order=True).head(n)
         .with_columns(pl.col("rid").cum_count().over("rid").alias("k")))
    wide = t.pivot(on="k", index="rid", values="tok")
    wide = wide.rename({str(k): f"{prefix}_{k}" for k in range(1, n + 1) if str(k) in wide.columns})
    for k in range(1, n + 1):
        if f"{prefix}_{k}" not in wide.columns:
            wide = wide.with_columns(pl.lit(None, pl.String).alias(f"{prefix}_{k}"))
    return wide


def add_keys(s1: pl.DataFrame, tg: pl.DataFrame):
    cols = ["country", "name_core", "name_ph", "name_ns", "addr_core", "first_num", "nums"]
    both = pl.concat([s1.select(cols).with_columns(pl.lit(0).alias("side")),
                      tg.select(cols).with_columns(pl.lit(1).alias("side"))]).with_row_index("rid")
    nt = _rare_tokens(both, "name_core", r"^[a-z0-9]{2,}$", N_RARE, "nt")
    at = _rare_tokens(both, "addr_core", r"^[a-z]{3,}$", N_RARE, "at")
    b = both.join(nt, on="rid", how="left").join(at, on="rid", how="left").sort("rid")
    keys = b.select("side", "country", **{k: e for k, (e, _, _) in _key_exprs().items()})
    k1 = keys.filter(pl.col("side") == 0).drop("side").with_row_index("i")
    kt = keys.filter(pl.col("side") == 1).drop("side").with_row_index("j")
    return k1, kt


def generate_candidates(s1: pl.DataFrame, tg: pl.DataFrame, cfg=None) -> pl.DataFrame:
    """Returns DataFrame(i, j, nkeys, keymask, bscore) of candidate index pairs.

    Runs one country at a time (keys never cross countries) to bound peak memory."""
    cfg = {**DEFAULT_BLOCKING, **(cfg or {})}
    out = []
    ct = tg["country"]
    for c in s1["country"].unique().sort():
        i_map = s1.with_row_index("gi").filter(pl.col("country") == c)
        j_map = tg.with_row_index("gj").filter(ct == c)
        if j_map.height == 0:
            continue
        cand = _candidates_one(i_map.drop("gi"), j_map.drop("gj"), cfg)
        cand = cand.with_columns(i=i_map["gi"].gather(cand["i"]), j=j_map["gj"].gather(cand["j"]))
        out.append(cand)
        del i_map, j_map
    cand = pl.concat(out) if out else pl.DataFrame(schema={"i": pl.UInt32, "j": pl.UInt32})
    return cand.with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))


def _candidates_one(s1: pl.DataFrame, tg: pl.DataFrame, cfg) -> pl.DataFrame:
    caps = {**KEY_CAPS, **cfg.get("caps", {})}
    assert len(caps) <= 32
    k1, kt = add_keys(s1, tg)
    cand = None
    for bit, (key, (cap1, capt)) in enumerate(caps.items()):
        def long(k, idx):
            return (k.select(idx, key).explode(key).drop_nulls().rename({key: "kv"}).unique())
        a, b = long(k1, "i"), long(kt, "j")
        n1 = a.group_by("kv").agg(pl.len().alias("n1"))
        nt = b.group_by("kv").agg(pl.len().alias("nt"))
        ok = (n1.join(nt, on="kv").filter((pl.col("n1") <= cap1) & (pl.col("nt") <= capt))
              .select("kv", w=1.0 / (1.0 + (pl.col("n1") * pl.col("nt")).cast(pl.Float32).log(2))))
        p = (a.join(ok, on="kv").join(b, on="kv")
             .group_by("i", "j").agg(pl.col("w").max())  # one vote per key type
             .select("i", "j", nkeys=pl.lit(1, pl.UInt8), keymask=pl.lit(1 << bit, pl.UInt32),
                     bscore=pl.col("w").cast(pl.Float32)))
        del a, b, n1, nt, ok
        cand = p if cand is None else (
            pl.concat([cand, p]).group_by("i", "j")
            .agg(pl.col("nkeys").sum().cast(pl.UInt8), pl.col("keymask").sum().cast(pl.UInt32),
                 pl.col("bscore").sum().cast(pl.Float32)))
    del k1, kt
    # cap per S1: keep the candidates with the most specific shared evidence
    return (cand.with_columns(pl.col("bscore").cast(pl.Float32))
            .sort(["i", "bscore", "j"], descending=[False, True, False])
            .group_by("i", maintain_order=True).head(cfg["max_per_s1"]))


def blocking_recall(cand_pairs: pl.DataFrame, truth_pairs: pl.DataFrame) -> float:
    """Both frames with columns (i, j) as UInt32."""
    hit = truth_pairs.join(cand_pairs.select("i", "j"), on=["i", "j"], how="semi").height
    return hit / max(truth_pairs.height, 1)
