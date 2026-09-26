"""Score a matching_results.tsv against a ground-truth file with macro F0.5.

    python src/evaluate.py --pred output/matching_results.tsv --truth path/to/ground_truth.tsv [--source1 test_source1.tsv]
"""
import argparse
import pandas as pd



def f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def read_lists(path, col):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {r.source1_entity_id: {x for x in getattr(r, col).split(",") if x} for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--source1", help="optional source1 tsv for a per-country breakdown")
    a = ap.parse_args()
    pred = read_lists(a.pred, "matched_entity_ids")
    truth = read_lists(a.truth, "matched_entity_ids")
    scores = {s: f05(pred.get(s, set()), t) for s, t in truth.items()}
    print(f"macro F0.5 = {sum(scores.values()) / len(scores):.4f}  (n={len(scores)})")
    if a.source1:
        s1 = pd.read_csv(a.source1, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        for c, ids in s1.groupby("country")["entity_id"]:
            v = [scores[i] for i in ids if i in scores]
            print(f"  {c}: {sum(v) / len(v):.4f} (n={len(v)})")


if __name__ == "__main__":
    main()
