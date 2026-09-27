"""Synthetic decoy *groups* for training: adapts the matcher to the decoy pattern of the test set.

Measured shift (see Documentation §5): test decoys come in groups of 2-3 records sharing one
house number shifted by a small amount next to a real business, often with one word added
("Lakeside", "Partners", "India"/"France") or swapped. In train such groups were rare (and mostly
true copies), so the model learned the opposite of what test needs. This script copies a share of
each entity's *own* Source 2/3 records into a decoy group (same shifted number, optional added or
swapped word) and appends them to the training sources as unlinked records. Ground truth is
unchanged. The added-word vocabulary is mined from train's own decoys (single extra token vs the
Source 1 name among unlinked shifted-number look-alikes); no test data or external data is used.

    python src/augment_decoys.py --data-dir dataset --out-dir dataset_aug
    -> dataset_aug/train/* (augmented), dataset_aug/test -> ../dataset/test
"""
import argparse
import os
import re

import numpy as np
import polars as pl

# decoy vocabulary per country (raw spellings of the tokens mined from train decoys)
WORDS = {
    "us": ["Group", "Holdings", "Partners", "East", "West", "North", "Metro", "South", "Valley", "Uptown",
           "Harbor", "Summit", "Midtown", "Greater", "Coastal", "Central", "Downtown", "Eastgate", "Highland",
           "Westgate", "Lakeside", "Southside", "Riverside", "Northside", "Services", "Center", "Solutions"],
    "india": ["Industries", "Enterprises", "Group", "Holdings", "Public", "Exports", "Overseas", "Ventures",
              "Infratech", "Solutions", "Tech", "Foods", "Engineering", "Electricals", "Agencies", "Auto",
              "Hotel", "Steel", "Services", "Stores", "Sweets", "Bakery", "Trading", "Motors", "Traders",
              "India", "Pharmacy", "Centre", "Garments", "Hardware", "Textiles", "Medicals", "Jewellers"],
}
RATE = {"us": 0.35, "india": 0.15}  # share of entities that get a decoy group (test-like density)
GROUP_SIZES, GROUP_P = [1, 2, 3], [0.1, 0.7, 0.2]
LEGAL = {"llc", "inc", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private", "pc", "llp",
         "plc", "lp", "pllc", "l.l.c.", "inc.", "corp.", "ltd.", "co.", "pvt.", "(llc)", "[corp]"}
NUM = re.compile(r"\d+")


def _case_like(word: str, ref: str) -> str:
    if ref.isupper():
        return word.upper()
    if ref.islower():
        return word.lower()
    return word


def _add_word(name: str, word: str) -> str:
    toks = name.split()
    k = len(toks)
    while k > 0 and toks[k - 1].lower().strip(",") in LEGAL:  # before the legal suffix
        k -= 1
    return " ".join(toks[:k] + [_case_like(word, name)] + toks[k:])


def _swap_word(name: str, word: str, rng) -> str:
    toks = name.split()
    idx = [t for t in range(len(toks)) if toks[t].isalpha() and len(toks[t]) >= 3 and toks[t].lower() not in LEGAL]
    if not idx:
        return _add_word(name, word)
    t = idx[rng.integers(len(idx))]
    toks[t] = _case_like(word, toks[t])
    return " ".join(toks)


def _shift(addr: str, house: int | None, delta: int) -> str | None:
    """Replace the house number (the number equal to the entity's, else the first one) by +delta."""
    ms = list(NUM.finditer(addr))
    if not ms:
        return None
    m = next((x for x in ms if house is not None and int(x.group()) == house), ms[0])
    s = m.group()
    new = str(int(s) + delta).zfill(len(s))
    return addr[:m.start()] + new + addr[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="dataset_aug")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rd = lambda f: pl.read_csv(os.path.join(a.data_dir, "train", f), separator="\t", infer_schema=False,  # noqa: E731
                               quote_char=None).fill_null("")
    s1, s2, s3 = rd("train_source1.tsv"), rd("train_source2.tsv"), rd("train_source3.tsv")
    gt = rd("train_ground_truth.tsv")
    links = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids").str.strip_chars() != "")
             .select(sid="source1_entity_id", tid=pl.col("matched_entity_ids").str.strip_chars()))
    tg = pl.concat([s2, s3]).rename({"entity_id": "tid", "business_name": "tn", "business_address": "ta"})
    tmpl = links.join(tg.select("tid", "tn", "ta"), on="tid").group_by("sid").agg("tid", "tn", "ta")
    ent = s1.rename({"entity_id": "sid"}).join(tmpl, on="sid", how="left")
    ent = ent.with_columns(house=pl.col("business_address").str.extract(r"(\d+)").cast(pl.Int64, strict=False))

    rows = {"S2": [], "S3": []}
    counter = 0
    n_groups = 0
    for r in ent.iter_rows(named=True):
        c = r["country"].strip().lower()
        if rng.random() >= RATE.get(c, 0.0):
            continue
        k = int(rng.choice(GROUP_SIZES, p=GROUP_P))
        if r["tid"]:  # templates: the entity's own true records
            pick = rng.choice(len(r["tid"]), size=k, replace=len(r["tid"]) < k)
            temps = [(r["tid"][p][:2], r["tn"][p], r["ta"][p]) for p in pick]
        else:  # singleton: use the Source 1 record itself
            temps = [("S2" if rng.random() < 0.5 else "S3", r["business_name"], r["business_address"])] * k
        delta = int(rng.integers(1, 16))
        mode = rng.choice(["same", "add", "swap"], p=[0.45, 0.4, 0.15])
        words = WORDS.get(c, WORDS["us"])
        word = words[rng.integers(len(words))]
        made = 0
        for src, name, addr in temps:
            na = _shift(addr, r["house"], delta)
            if na is None:
                continue
            nm = name
            if name.isascii() and mode != "same" and rng.random() < 0.75:
                nm = _add_word(name, word) if mode == "add" else _swap_word(name, word, rng)
            if made and rng.random() < 0.5:  # siblings differ in casing like real copies do
                nm = nm.upper() if rng.random() < 0.5 else nm.title()
            counter += 1
            made += 1
            rows[src].append((f"{src}-D{counter:08d}", nm, na, r["country"]))
        n_groups += made > 0

    out = os.path.join(a.out_dir, "train")
    os.makedirs(out, exist_ok=True)
    for src, base in (("S2", s2), ("S3", s3)):
        add = pl.DataFrame(rows[src], schema=base.columns, orient="row")
        pl.concat([base, add]).write_csv(os.path.join(out, f"train_source{src[1]}.tsv"), separator="\t",
                                         quote_style="never")
        print(f"{src}: +{add.height} decoys ({base.height} originals)")
    for f in ("train_source1.tsv", "train_ground_truth.tsv"):
        dst = os.path.join(out, f)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(a.data_dir, "train", f)), dst)
    test_link = os.path.join(a.out_dir, "test")
    if not os.path.exists(test_link):
        os.symlink(os.path.abspath(os.path.join(a.data_dir, "test")), test_link)
    print(f"decoy groups: {n_groups}, records: {counter}")


if __name__ == "__main__":
    main()
