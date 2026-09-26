"""Record preparation + multi-pass blocking (candidate generation)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import (acronym, canon_country, core_name, name_variants, norm_address, norm_name,
                       phonetic)

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[SOURCE_COLS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"id": df["entity_id"].str.strip().values})
    out["src"] = out["id"].str[:2]
    out["country"] = [canon_country(c) for c in df["country"]]
    out["name_raw"] = df["business_name"].values
    out["name_n"] = [norm_name(n) for n in df["business_name"]]
    out["name_vars"] = [name_variants(n) for n in df["business_name"]]
    out["name_core"] = [core_name(n) for n in out["name_n"]]
    out["name_ph"] = [" ".join(phonetic(t) for t in c.split()) for c in out["name_core"]]
    out["name_acr"] = [acronym(c) for c in out["name_core"]]
    addrs = [norm_address(a, c) for a, c in zip(df["business_address"], out["country"])]
    for k in ("full", "core", "landmark", "postal", "nums", "first_num"):
        out["addr_" + k] = [a[k] for a in addrs]
    out["all_text"] = out["name_core"] + " " + out["addr_core"]
    return out


class Vectors:
    """TF-IDF spaces fitted transductively on all records of a split (no labels used)."""

    SPACES = {
        "name_char": ("name_core", dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)),
        "name_ph": ("name_ph", dict(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True)),
        "name_word": ("name_core", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)),
        "addr_char": ("addr_core", dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        "addr_word": ("addr_core", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)),
        "all_word": ("all_text", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)),
    }

    def __init__(self, s1: pd.DataFrame, tg: pd.DataFrame):
        self.m1, self.mt = {}, {}
        for key, (col, kw) in self.SPACES.items():
            v = TfidfVectorizer(dtype=np.float32, **kw)
            v.fit(pd.concat([s1[col], tg[col]]).values)
            self.m1[key] = v.transform(s1[col].values).tocsr()
            self.mt[key] = v.transform(tg[col].values).tocsr()

    def pair_cos(self, key, i, j, chunk=200_000):
        a, b = self.m1[key], self.mt[key]
        out = np.empty(len(i), dtype=np.float32)
        for s in range(0, len(i), chunk):
            e = s + chunk
            out[s:e] = np.asarray(a[i[s:e]].multiply(b[j[s:e]]).sum(1)).ravel()
        return out


def _topk(A, B, k, a_idx, b_idx, budget=4e7):
    """Top-k cosine neighbours in B for every row of A (both L2-normalised sparse)."""
    if len(a_idx) == 0 or len(b_idx) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    A, Bt = A[a_idx], B[b_idx].T.tocsc()
    k = min(k, len(b_idx))
    step = max(1, int(budget // max(len(b_idx), 1)))
    ri, ci, sc = [], [], []
    for s in range(0, len(a_idx), step):
        S = (A[s:s + step] @ Bt).toarray()
        top = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < S.shape[1] else np.tile(np.arange(S.shape[1]), (S.shape[0], 1))
        vals = np.take_along_axis(S, top, 1)
        keep = vals > 0
        rows = np.repeat(np.arange(s, s + S.shape[0]), top.shape[1]).reshape(top.shape)
        ri.append(a_idx[rows[keep]]); ci.append(b_idx[top[keep]]); sc.append(vals[keep])
    return np.concatenate(ri), np.concatenate(ci), np.concatenate(sc)


DEFAULT_BLOCKING = dict(
    k_name=25, k_ph=15, k_addr=10, k_all=25, k_global=5, k_rev=3, max_per_s1=60,
)


def generate_candidates(s1, tg, vec: Vectors, cfg=None) -> pd.DataFrame:
    """Union of several blocking passes; returns DataFrame(i, j) of index pairs."""
    cfg = {**DEFAULT_BLOCKING, **(cfg or {})}
    parts = []
    c1 = s1["country"].values
    ct = tg["country"].values
    all_t = np.arange(len(tg))
    for c in pd.unique(c1):
        a_idx = np.where(c1 == c)[0]
        b_idx = np.where(ct == c)[0]
        if len(b_idx) == 0:  # unseen/mismatched country label -> search everything
            b_idx = all_t
        for key, k in (("name_char", cfg["k_name"]), ("name_ph", cfg["k_ph"]),
                       ("addr_char", cfg["k_addr"]), ("all_word", cfg["k_all"])):
            i, j, _ = _topk(vec.m1[key], vec.mt[key], k, a_idx, b_idx)
            parts.append((i, j))
        # reverse pass: every target gets its best S1 entities in the same country
        j, i, _ = _topk(vec.mt["all_word"], vec.m1["all_word"], cfg["k_rev"], b_idx, a_idx)
        parts.append((i, j))
        j, i, _ = _topk(vec.mt["name_char"], vec.m1["name_char"], cfg["k_rev"], b_idx, a_idx)
        parts.append((i, j))
    # global name pass (country labels can be noisy / open-set)
    i, j, _ = _topk(vec.m1["name_char"], vec.mt["name_char"], cfg["k_global"], np.arange(len(s1)), all_t)
    parts.append((i, j))
    # exact postal + first name token block
    parts.append(_postal_block(s1, tg))

    i = np.concatenate([p[0] for p in parts]); j = np.concatenate([p[1] for p in parts])
    cand = pd.DataFrame({"i": i, "j": j}).drop_duplicates().reset_index(drop=True)

    # cap per S1 by a cheap blocking score; keep reverse-best pairs regardless
    cheap = (0.45 * vec.pair_cos("name_char", cand.i.values, cand.j.values)
             + 0.2 * vec.pair_cos("name_ph", cand.i.values, cand.j.values)
             + 0.35 * vec.pair_cos("all_word", cand.i.values, cand.j.values))
    cand["cheap"] = cheap
    cand["r1"] = cand.groupby("i")["cheap"].rank(ascending=False, method="first")
    cand["rt"] = cand.groupby("j")["cheap"].rank(ascending=False, method="first")
    cand = cand[(cand.r1 <= cfg["max_per_s1"]) | (cand.rt <= 2)]
    return cand[["i", "j"]].reset_index(drop=True)


def _postal_block(s1, tg, max_block=200):
    t = tg[tg.addr_postal != ""].assign(j=lambda d: d.index.values)
    t["ft"] = [c.split()[0] if c else "" for c in t.name_core]
    a = s1[s1.addr_postal != ""].assign(i=lambda d: d.index.values)
    a["ft"] = [c.split()[0] if c else "" for c in a.name_core]
    m = a[["i", "addr_postal", "ft"]].merge(t[["j", "addr_postal", "ft"]], on=["addr_postal", "ft"])
    m = m[m.groupby("i")["j"].transform("size") <= max_block]
    return m.i.values, m.j.values


def blocking_recall(cand_ids: dict, truth: dict) -> float:
    tot = hit = 0
    for s, tr in truth.items():
        tot += len(tr)
        hit += len(tr & cand_ids.get(s, set()))
    return hit / max(tot, 1)
