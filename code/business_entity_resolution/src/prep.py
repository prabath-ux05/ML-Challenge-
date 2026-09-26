"""Parallel record normalisation, cached to parquet (one file per split)."""
import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from normalize import (acronym, canon_country, core_name, name_variants, norm_address, norm_name,
                       phonetic)

COLS = ["id", "country", "name_n", "name_core", "name_ph", "name_ns", "name_acr", "name_vars",
        "addr_core", "addr_full", "has_landmark", "postal", "first_num", "nums"]


def read_source(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    need = ["entity_id", "business_name", "business_address", "country"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df.select(need).with_columns(pl.all().fill_null(""))


def _norm_rows(rows):
    out = []
    for eid, name, addr, country in rows:
        c = canon_country(country)
        nn = norm_name(name)
        core = core_name(nn)
        ad = norm_address(addr, c)
        out.append((
            eid.strip(), c, nn, core,
            " ".join(phonetic(t) for t in core.split()),
            core.replace(" ", ""),
            acronym(core),
            " | ".join(name_variants(name)),
            ad["core"], ad["full"], bool(ad["landmark"]), ad["postal"], ad["first_num"], " ".join(ad["nums"]),
        ))
    return out


def normalise(df: pl.DataFrame, workers: int, chunk: int = 50_000) -> pl.DataFrame:
    rows = df.rows()
    parts = [rows[s:s + chunk] for s in range(0, len(rows), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_norm_rows, parts))
    flat = [r for part in res for r in part]
    return pl.DataFrame(flat, schema=COLS, orient="row")


def load_split(data_dir: str, split: str, cache_dir: str, workers: int):
    """Return (s1, tg) normalised frames; tg = Source 2 + Source 3 concatenated."""
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for name, srcs in (("s1", (1,)), ("tg", (2, 3))):
        cp = os.path.join(cache_dir, f"{split}_{name}.parquet")
        if os.path.exists(cp):
            out.append(pl.read_parquet(cp))
            continue
        raw = pl.concat([read_source(os.path.join(data_dir, split, f"{split}_source{k}.tsv")) for k in srcs])
        df = normalise(raw, workers)
        df.write_parquet(cp)
        out.append(df)
    return out[0], out[1]
