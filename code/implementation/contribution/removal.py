# Πείραμα αφαίρεσης. Βελτιώνεται το μοντέλο αν δεν συμμετέχουν οι clients με αρνητικό φ;
#
# Αρνητικό φ σημαίνει βλάβη κατά μέσο όρο σε όλους τους συνασπισμούς. Δεν εγγυάται
# ότι το μοντέλο βελτιώνεται αν αυτοί αφαιρεθούν από τους 21. Ο έλεγχος γίνεται με
# επανεκπαίδευση χωρίς αυτούς (Ghorbani και Zou 2019, Σχ. 3).
#
# Οι τιμές φ μετρήθηκαν στο ίδιο test. Αν οι αφαιρούμενοι επιλέγονταν και μετριόταν το
# κέρδος στα ίδια άτομα, το κέρδος θα φούσκωνε. Γι' αυτό τα 9 άτομα του test χωρίζονται
# σε 3 σταθερές ομάδες. Σε κάθε ομάδα k, οι αφαιρούμενοι επιλέγονται με το φ ανά άτομο
# (contribution_classwise.py --value_metric per_subject) στα 6 άτομα εκτός ομάδας (A_k).
# Το αποτέλεσμα μετριέται στα 3 της ομάδας (B_k).
#
# Οι συνθήκες κάθε seed είναι όλοι οι clients, χωρίς τους αρνητικούς (R_k), χωρίς τους m
# χαμηλότερους ή τους m υψηλότερους (m = 3, 6, 9, 12) και χωρίς τυχαίο σύνολο ίσου
# μεγέθους με το R_k. Κάθε συνθήκη είναι μία ομοσπονδιακή εκπαίδευση 100 γύρων.
#
#   python -m implementation.contribution.removal --seeds 0 1 2 --out results/contribution/removal_parts/part0.json
#   python -m implementation.contribution.removal --analyze results/contribution/removal_parts/*.json
#
# Οι προβλέψεις P12-P16 στο analyze() γράφτηκαν πριν από το τρέξιμο. Το
# αποτέλεσμα καταγράφει αν επιβεβαιώθηκαν.

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

from implementation.common.helpers import resolve_device
from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.dataset.export_processed import common_provenance, load_processed, provenance_block
from implementation.federated.config import federated_args
from implementation.federated.federated import run_one

FOLDS = 3
M_STEPS = (3, 6, 9, 12)
PER_SUBJECT = "results/contribution/har_contribution_per_subject.json"
HEADLINE_PHI = "results/contribution/har_contribution_ann_20seeds.json"
FED_RESULT = "results/federated/har_fed_ann.json"


def folds_of(labels):
    ids = sorted(labels)
    return [ids[k::FOLDS] for k in range(FOLDS)]


def valuation(ps, subjects, estimator="kernelshap_cw"):
    """φ_i στα άτομα S: Σ_{s∈S} w_s φ_{i,s} / Σ_{s∈S} w_s, με τον μέσο των 20 seeds."""
    labels = ps["per_seed"][0]["output_labels"]
    phi = np.asarray(ps["phi"][estimator])
    w = np.asarray(ps["class_weights"])
    idx = [labels.index(s) for s in subjects]
    return phi[:, idx] @ w[idx] / w[idx].sum()


def plan_conditions(ps):
    """Οι συνθήκες που είναι ίδιες σε όλα τα seeds. Οι τυχαίες φτιάχνονται ανά seed."""
    labels = ps["per_seed"][0]["output_labels"]
    conds, meta = {"all": ()}, {}
    for k, B in enumerate(folds_of(labels)):
        A = [s for s in labels if s not in B]
        phiA = valuation(ps, A)
        order = [int(i) for i in np.argsort(phiA, kind="stable")]
        R = tuple(sorted(int(i) for i in np.where(phiA < 0)[0]))
        conds[f"neg_{k}"] = R
        for m in M_STEPS:
            conds[f"low_{k}_{m}"] = tuple(sorted(order[:m]))
            conds[f"high_{k}_{m}"] = tuple(sorted(order[-m:]))
        meta[k] = {"B": B, "A": A, "R": list(R), "phiA": phiA.tolist(), "order": order}
    return conds, meta


def random_set(k, size, seed, n_clients):
    rng = np.random.default_rng(10_000 + 100 * k + seed)
    return tuple(sorted(int(i) for i in rng.choice(n_clients, size=size, replace=False)))


def run_seed(fargs, trainset, testset, client_ids, conds, meta, seed, head):
    n_clients = len(np.unique(client_ids))
    seed_conds = dict(conds)
    for k in meta:
        seed_conds[f"rand_{k}"] = random_set(k, len(meta[k]["R"]), seed, n_clients)
    cache, rows = {}, {}
    for name, excl in seed_conds.items():
        if excl not in cache:        # ίδιο σύνολο αφαίρεσης, ίδιο τρέξιμο
            a = copy.deepcopy(fargs)
            a.selector, a.exclude = ("all", None) if not excl else ("exclude", list(excl))
            t = time.time()
            r = run_one(a, trainset, testset, client_ids, seed)
            cache[excl] = {"final_accuracy": r["final_accuracy"],
                           "per_test_subject_accuracy": r["per_test_subject_accuracy"],
                           "per_test_subject_n": r["per_test_subject_n"],
                           "seconds": round(time.time() - t, 1)}
        rows[name] = {"exclude": list(excl), **cache[excl]}
    # η συνθήκη "all" πρέπει να είναι το μοντέλο της ομοσπονδιακής εκπαίδευσης
    ok = None if head is None else abs(rows["all"]["final_accuracy"] - head) <= 1e-12
    print(f"  seed {seed}: {len(cache)} τρεξίματα, με όλους accuracy {rows['all']['final_accuracy']:.6f}, "
          f"ίδιο με την ομοσπονδιακή: {ok}", flush=True)
    return {"seed": seed, "matched_headline": ok, "n_fl_runs": len(cache), "conditions": rows}


def acc_on(row, subjects):
    """Accuracy στα άτομα subjects, από το accuracy και το πλήθος samples κάθε ατόμου."""
    a, n = row["per_test_subject_accuracy"], row["per_test_subject_n"]
    num = sum(a[str(s)] * n[str(s)] if str(s) in a else a[s] * n[s] for s in subjects)
    den = sum(n[str(s)] if str(s) in n else n[s] for s in subjects)
    return num / den


def analyze(paths):
    parts = [json.load(open(p, encoding="utf-8")) for p in paths]
    seeds = sorted((s for p in parts for s in p["seeds"]), key=lambda s: s["seed"])
    meta = {int(k): v for k, v in parts[0]["meta"].items()}
    head9 = set(int(i) for i in np.where(np.asarray(
        json.load(open(HEADLINE_PHI))["summary"]["per_estimator"]["kernelshap"]["phi_mean"]) < 0)[0])
    res = {"n_seeds": len(seeds), "all_matched_headline": all(s["matched_headline"] is True for s in seeds)}
    for k, mk in meta.items():
        A, B, R = mk["A"], mk["B"], set(mk["R"])
        d = lambda c1, c2, S: np.array([acc_on(s["conditions"][c1], S) - acc_on(s["conditions"][c2], S)
                                        for s in seeds])
        neg_B, neg_A, rnd_B = d(f"neg_{k}", "all", B), d(f"neg_{k}", "all", A), d(f"neg_{k}", f"rand_{k}", B)
        curve = {m: {"low": float(np.mean([acc_on(s["conditions"][f"low_{k}_{m}"], B) for s in seeds])),
                     "high": float(np.mean([acc_on(s["conditions"][f"high_{k}_{m}"], B) for s in seeds]))}
                 for m in M_STEPS}
        stat = lambda v: {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "seeds_pos": int((v > 0).sum())}
        res[f"fold_{k}"] = {
            "B": B, "A": A, "R": sorted(R), "size_R": len(R),
            # επικάλυψη του R_k με τους 9 αρνητικούς του βαθμωτού φ (Jaccard)
            "jaccard_with_headline_9": len(R & head9) / len(R | head9) if (R | head9) else 1.0,
            "acc_B_all": float(np.mean([acc_on(s["conditions"]["all"], B) for s in seeds])),
            "neg_minus_all_B": stat(neg_B), "neg_minus_all_A": stat(neg_A),
            "neg_minus_rand_B": stat(rnd_B), "curve_B": curve}
    f = [res[f"fold_{k}"] for k in meta]
    p14 = [all(x["curve_B"][m]["low"] >= x["curve_B"][m]["high"] for m in M_STEPS) for x in f]
    p16 = [x["neg_minus_all_A"]["mean"] > x["neg_minus_all_B"]["mean"] for x in f]
    res["predictions"] = {
        "P12": {"folds_pos": sum(x["neg_minus_all_B"]["mean"] > 0 for x in f),
                "pass": sum(x["neg_minus_all_B"]["mean"] > 0 for x in f) >= 2},
        "P13": {"folds_pos": sum(x["neg_minus_rand_B"]["mean"] > 0 for x in f),
                "pass": sum(x["neg_minus_rand_B"]["mean"] > 0 for x in f) >= 2},
        "P14": {"folds_ok": sum(p14), "pass": all(p14)},
        "P15": {"jaccards": [x["jaccard_with_headline_9"] for x in f],
                "pass": all(x["jaccard_with_headline_9"] >= 0.6 for x in f)},
        "P16": {"folds_A_gt_B": sum(p16), "pass": sum(p16) >= 2}}
    res["data_provenance"] = common_provenance([p.get("data_provenance") for p in parts])
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="Πείραμα αφαίρεσης clients.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--source", default=PER_SUBJECT)
    p.add_argument("--fl_rounds", type=int, default=None, help="μόνο για γρήγορη δοκιμή")
    p.add_argument("--analyze", nargs="+", default=None)
    p.add_argument("--out", default=None)
    add_overwrite_flag(p)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = ("results/contribution/har_removal.json" if args.analyze
                    else "results/contribution/removal_parts/part.json")
    check_output(args.out, args.overwrite)

    if args.analyze:
        res = analyze(args.analyze)
        for k in range(FOLDS):
            x = res[f"fold_{k}"]
            print(f"\nομάδα {k}: B = {x['B']}, |R| = {x['size_R']}, Jaccard με τους 9 {x['jaccard_with_headline_9']:.2f}")
            for key in ("neg_minus_all_B", "neg_minus_all_A", "neg_minus_rand_B"):
                y = x[key]
                print(f"  {key:<18} {100 * y['mean']:+.2f} ± {100 * y['std']:.2f} μονάδες "
                      f"(θετικό σε {y['seeds_pos']}/{res['n_seeds']} seeds)")
        for k, y in res["predictions"].items():
            print(f"  {k}: {'επιβεβαιώθηκε' if y['pass'] else 'δεν επιβεβαιώθηκε'}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"experiment": "removal_analysis", "parts": args.analyze, **res}, fh, indent=2, ensure_ascii=False)
        print(f"γράφτηκε το {args.out}")
        return 0

    conds, meta = plan_conditions(json.load(open(args.source, encoding="utf-8")))
    fargs = federated_args([])
    if args.fl_rounds is not None:
        fargs.fl_rounds = args.fl_rounds
    fargs.device = resolve_device(fargs.device)
    trainset, testset, client_ids, prov = load_processed()
    head = {}
    if args.fl_rounds is None and os.path.exists(FED_RESULT):
        head = {r["seed"]: r["final_accuracy"]
                for r in json.load(open(FED_RESULT))["results"]["subject"]["per_seed"]}
    for k, mk in meta.items():
        print(f"ομάδα {k}: B = {mk['B']}, R = {mk['R']}")
    runs = [run_seed(fargs, trainset, testset, client_ids, conds, meta, sd, head.get(sd)) for sd in args.seeds]
    out = {"experiment": "removal_runs", "source": args.source, "folds": FOLDS, "m_steps": list(M_STEPS),
           "fl_args": vars(fargs), "meta": {str(k): v for k, v in meta.items()},
           "fixed_conditions": {k: list(v) for k, v in conds.items()}, "seeds": runs,
           "data_provenance": provenance_block(prov)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
