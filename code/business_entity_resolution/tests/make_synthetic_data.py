"""Generate a synthetic dataset with the same schema/noise patterns as the challenge.

Only used to smoke-test the pipeline when the real data is not present.
Usage: python tests/make_synthetic_data.py --out /path/to/dataset --n 3000
"""
import argparse
import os
import random

import pandas as pd

W = {
    "US": dict(
        first=["Acme", "Blue Ridge", "Golden Gate", "Liberty", "Summit", "Pioneer", "Eagle", "Pacific",
               "Atlantic", "Evergreen", "Silver Oak", "Red Rock", "Maple", "Hudson", "Lakeside", "Northstar",
               "Ironwood", "Cedar", "Keystone", "Frontier", "Johnson", "Miller", "Anderson", "Walker"],
        kind=["Plumbing", "Dental", "Bakery", "Auto Repair", "Consulting", "Logistics", "Software",
              "Construction", "Insurance", "Realty", "Pharmacy", "Coffee", "Hardware", "Printing", "Law"],
        legal=["Inc", "LLC", "Corp", "Corporation", "Co", "Company", "Incorporated", "Ltd", ""],
        streets=["Main", "Oak", "Maple", "Washington", "Park", "Elm", "Lake", "Hill", "Sunset", "Broadway",
                 "Market", "Pine", "Cedar", "Highland", "River"],
        stype=["Street", "Avenue", "Road", "Boulevard", "Drive", "Lane"],
        cities=[("Austin", "Texas", "787"), ("Denver", "Colorado", "802"), ("Seattle", "Washington", "981"),
                ("Boston", "Massachusetts", "021"), ("Chicago", "Illinois", "606"),
                ("Phoenix", "Arizona", "850"), ("Miami", "Florida", "331"), ("Portland", "Oregon", "972")],
    ),
    "India": dict(
        first=["Sharma", "Shree Ganesh", "Lakshmi", "Balaji", "Sai", "Krishna", "Patel", "Gupta", "Om Sai",
               "Srinivasa", "Annapurna", "Mahalaxmi", "Venkateshwara", "Reddy", "Agarwal", "Jain", "Tata",
               "Bharat", "Hindustan", "Sri Durga", "Murugan", "Kaveri", "Ganga", "Royal"],
        kind=["Traders", "Enterprises", "Textiles", "Steels", "Electricals", "Pharma", "Motors", "Jewellers",
              "Sweets", "Agencies", "Builders", "Hardware", "Medicals", "Infotech", "Foods"],
        legal=["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "Ltd", "Limited", "LLP", "& Sons", "& Co", ""],
        streets=["MG", "Station", "Nehru", "Gandhi", "Tilak", "Subhash", "Rajaji", "Ring", "Temple", "Market"],
        stype=["Road", "Marg", "Street", "Cross Road", "Main Road"],
        cities=[("Mumbai", "Maharashtra", "400"), ("Bengaluru", "Karnataka", "560"), ("Chennai", "Tamil Nadu", "600"),
                ("Kolkata", "West Bengal", "700"), ("Pune", "Maharashtra", "411"), ("Hyderabad", "Telangana", "500"),
                ("Gurugram", "Haryana", "122"), ("Jaipur", "Rajasthan", "302")],
    ),
    "France": dict(
        first=["Dupont", "Martin", "Bernard", "Le Petit", "Boulangerie", "Maison", "Atelier", "Lefebvre",
               "Moreau", "Garnier", "Rousseau", "Fontaine", "Chevalier", "Bonnet", "Lambert", "Mercier"],
        kind=["Boulangerie", "Transports", "Conseil", "Immobilier", "Pharmacie", "Plomberie", "Informatique",
              "Garage", "Bistrot", "Coiffure", "Menuiserie", "Assurances"],
        legal=["SARL", "SAS", "SA", "EURL", "S.A.R.L.", "SASU", ""],
        streets=["de la Paix", "Victor Hugo", "de la Republique", "Jean Jaures", "du Général de Gaulle",
                 "Pasteur", "des Lilas", "Saint-Michel", "de Verdun", "Gambetta"],
        stype=["Rue", "Avenue", "Boulevard", "Place", "Chemin"],
        cities=[("Paris", "", "750"), ("Lyon", "", "690"), ("Marseille", "", "130"), ("Toulouse", "", "310"),
                ("Nice", "", "060"), ("Nantes", "", "440"), ("Bordeaux", "", "330"), ("Lille", "", "590")],
    ),
}
LANDMARKS = ["Near SBI ATM", "Opp. Bus Stand", "Behind City Mall", "Near Railway Station", "Opposite HDFC Bank"]
ABBR = {"Street": "St", "Avenue": "Ave", "Road": "Rd", "Boulevard": "Blvd", "Drive": "Dr", "Lane": "Ln",
        "Private": "Pvt", "Limited": "Ltd", "Corporation": "Corp", "Company": "Co", "Incorporated": "Inc",
        "Rue": "R.", "Boulevard ": "Bd ", "Marg": "Marg", "Enterprises": "Ent", "Traders": "Trdrs",
        "Bengaluru": "Bangalore", "Mumbai": "Bombay", "Chennai": "Madras", "Gurugram": "Gurgaon", "&": "and",
        "Saint": "St", "Maharashtra": "MH", "Karnataka": "KA", "Texas": "TX", "Colorado": "CO",
        "Washington": "WA", "Illinois": "IL"}


def typo(s, r):
    if len(s) < 4:
        return s
    i = r.randrange(1, len(s) - 1)
    op = r.random()
    if op < 0.33:
        return s[:i] + s[i + 1:]
    if op < 0.66:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + r.choice("aeioustrnl") + s[i + 1:]


def make_entity(country, r):
    w = W[country]
    name = f"{r.choice(w['first'])} {r.choice(w['kind'])}"
    legal = r.choice(w["legal"])
    city, state, pfx = r.choice(w["cities"])
    num = r.randint(1, 999)
    if country == "France":
        addr = f"{num} {r.choice(w['stype'])} {r.choice(w['streets'])}, {pfx}{r.randint(0, 20):02d} {city}"
    elif country == "India":
        addr = (f"{r.choice(['Plot No.', 'Shop No.', 'No.', '#', ''])} {num}, {r.choice(w['streets'])} "
                f"{r.choice(w['stype'])}, {city}, {state} {pfx}{r.randint(0, 99):03d}")
    else:
        addr = f"{num} {r.choice(w['streets'])} {r.choice(w['stype'])}, {city}, {state} {pfx}{r.randint(0, 99):02d}"
    return {"name": f"{name} {legal}".strip(), "addr": addr, "country": country}


def perturb(e, r, strength=1.0):
    name, addr = e["name"], e["addr"]
    for k, v in ABBR.items():
        if r.random() < 0.5 * strength:
            if k in name:
                name = name.replace(k, v)
            elif v in name and r.random() < 0.3:
                name = name.replace(v, k)
            if k in addr:
                addr = addr.replace(k, v)
    if r.random() < 0.3 * strength:
        name = typo(name, r)
    if r.random() < 0.2 * strength:
        addr = typo(addr, r)
    if r.random() < 0.15 * strength:
        toks = name.split()
        if len(toks) > 1:
            i = r.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
            name = " ".join(toks)
    if r.random() < 0.25 * strength:  # drop legal suffix
        name = " ".join(name.split()[:2])
    if r.random() < 0.3 * strength:  # drop postal / state
        parts = addr.split(",")
        addr = ",".join(parts[:-1]) if len(parts) > 1 else addr
    if e["country"] == "India" and r.random() < 0.3 * strength:
        parts = addr.split(",")
        parts.insert(1, " " + r.choice(LANDMARKS))
        addr = ",".join(parts)
    if r.random() < 0.1 * strength:
        name = name + " dba " + name.split()[0] + " Shop"
    if r.random() < 0.2:
        name = name.upper() if r.random() < 0.5 else name.lower()
    return {"name": name, "addr": addr, "country": e["country"]}


def generate(n, countries, seed):
    r = random.Random(seed)
    ents = []
    for i in range(n):
        c = r.choice(countries)
        e = make_entity(c, r)
        ents.append(e)
        if r.random() < 0.15:  # chain branch: same name, different address -> different entity
            b = make_entity(c, r)
            b["name"] = e["name"]
            ents.append(b)
    r.shuffle(ents)
    s1, s2, s3, gt = [], [], [], []
    distract = []
    for i, e in enumerate(ents):
        sid = f"S1-{i + 1:05d}"
        base = perturb(e, r, 0.3)
        s1.append((sid, base["name"], base["addr"], e["country"]))
        matches = []
        roll = r.random()
        n2 = 0 if roll < 0.3 else (1 if roll < 0.8 else r.randint(2, 3))
        n3 = 0 if r.random() < 0.4 else (1 if r.random() < 0.8 else 2)
        for _ in range(n2):
            matches.append(("S2", perturb(e, r)))
        for _ in range(n3):
            matches.append(("S3", perturb(e, r)))
        gt.append([sid, matches])
    for i in range(int(len(ents) * 0.3)):  # unmatched records in S2/S3
        distract.append((r.choice(["S2", "S3"]), perturb(make_entity(r.choice(countries), r), r)))
    pool = [(g, src, rec) for g in range(len(gt)) for src, rec in gt[g][1]] + [(None, s, x) for s, x in distract]
    r.shuffle(pool)
    c2 = c3 = 0
    ids = {i: [] for i in range(len(gt))}
    for g, src, rec in pool:
        if src == "S2":
            c2 += 1
            eid = f"S2-{c2:05d}"
            s2.append((eid, rec["name"], rec["addr"], rec["country"]))
        else:
            c3 += 1
            eid = f"S3-{c3:05d}"
            s3.append((eid, rec["name"], rec["addr"], rec["country"]))
        if g is not None:
            ids[g].append(eid)
    cols = ["entity_id", "business_name", "business_address", "country"]
    gt_df = pd.DataFrame({"source1_entity_id": [g[0] for g in gt],
                          "matched_entity_ids": [",".join(ids[i]) for i in range(len(gt))]})
    return pd.DataFrame(s1, columns=cols), pd.DataFrame(s2, columns=cols), pd.DataFrame(s3, columns=cols), gt_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    for split, countries, n, seed in [("train", ["US", "India"], a.n, 1), ("test", ["US", "India", "France"], a.n // 2, 2)]:
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        s1, s2, s3, gt = generate(n, countries, seed)
        for k, df in [("source1", s1), ("source2", s2), ("source3", s3)]:
            df.to_csv(os.path.join(d, f"{split}_{k}.tsv"), sep="\t", index=False)
        gt.to_csv(os.path.join(d, f"{split}_ground_truth.tsv"), sep="\t", index=False)
        print(split, len(s1), len(s2), len(s3))


if __name__ == "__main__":
    main()
