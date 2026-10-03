# Πότε βλάπτουν οι clients με αρνητικό φ. Το φ κάθε client χωρίζεται σε ομάδες γύρων.
#
# Το φ είναι άθροισμα των τιμών ανά γύρο, οπότε μπορεί να χωριστεί σε ομάδες γύρων για
# να φανεί αν η βλάβη είναι ομοιόμορφη ή συγκεντρωμένη σε μια φάση της εκπαίδευσης.
# Δεν εκπαιδεύει τίποτα. Διαβάζει τις τιμές ανά γύρο του contribution.py (20 seeds,
# KernelSHAP και GTG-Tib). Δύο χωρισμοί των γύρων, ορισμένοι και οι δύο από πριν:
#   checkpoints  1-10, 11-25, 26-50, 51-100 (τα όρια των --checkpoints)
#   quarters     1-25, 26-50, 51-75, 76-100
# Οι προβλέψεις P1-P6 γράφτηκαν πριν από την ανάλυση.
#
#   python -m implementation.contribution.temporal_profile

import argparse
import json
import os
import sys

import numpy as np

from implementation.common.outputs import add_overwrite_flag, check_output

SCHEMES = {"checkpoints": [(1, 10), (11, 25), (26, 50), (51, 100)],
           "quarters": [(1, 25), (26, 50), (51, 75), (76, 100)]}
ESTIMATORS = ("kernelshap", "gtg")


def bucket_sums(per_round, n_clients, bucket):
    """Άθροισμα φ ανά client στους γύρους του bucket και πλήθος βαθμολογημένων γύρων."""
    lo, hi = bucket
    rows = [np.asarray(r["phi"]) for r in per_round if lo <= r["round"] <= hi]
    if not rows:
        return np.zeros(n_clients), 0, np.zeros((0, n_clients))
    mat = np.vstack(rows)
    return mat.sum(axis=0), len(rows), mat


def profile(data, estimator, scheme, neg):
    seeds = data["per_seed"]
    n = len(data["summary"]["clients"])
    pos = [i for i in range(n) if i not in neg]
    out = []
    for b in SCHEMES[scheme]:
        S, scored, skipped, neg_frac_num, neg_frac_den = [], [], [], np.zeros(n), 0
        for s in seeds:
            sums, k, mat = bucket_sums(s["shapley"][estimator]["per_round"], n, b)
            S.append(sums)
            scored.append(k)
            skipped.append((b[1] - b[0] + 1) - k)
            neg_frac_num += (mat < 0).sum(axis=0)
            neg_frac_den += mat.shape[0]
        S = np.asarray(S)
        neg_g, pos_g = S[:, neg].sum(1), S[:, pos].sum(1)
        net = S.sum(1)                   # η μεταβολή του accuracy στους βαθμολογημένους γύρους
        gross = np.abs(S).sum(1)         # πόση κίνηση υπάρχει πίσω από αυτή τη μεταβολή
        out.append({
            "rounds": list(b),
            "scored_rounds_mean": float(np.mean(scored)),
            "skipped_rounds_mean": float(np.mean(skipped)),
            "per_client_sum_mean": S.mean(0).tolist(),
            "per_client_sum_std": S.std(0, ddof=1).tolist(),
            "per_client_neg_round_fraction": (neg_frac_num / max(neg_frac_den, 1)).tolist(),
            "neg_group_sum_mean": float(neg_g.mean()), "neg_group_sum_std": float(neg_g.std(ddof=1)),
            # οι ομάδες έχουν διαφορετικό πλήθος γύρων, οπότε συγκρίνεται ο ρυθμός ανά γύρο
            "neg_group_per_scored_round": float(neg_g.sum() / max(np.sum(scored), 1)),
            "pos_group_per_scored_round": float(pos_g.sum() / max(np.sum(scored), 1)),
            "pos_group_sum_mean": float(pos_g.mean()), "pos_group_sum_std": float(pos_g.std(ddof=1)),
            "net_mean": float(net.mean()), "gross_mean": float(gross.mean()),
            "seeds_neg_group_below_zero": int((neg_g < 0).sum()),
            "seeds_neg_group_at_or_above_zero": int((neg_g >= 0).sum()),
            "seeds_gross_ge_10x_net": int((gross >= 10 * np.abs(net)).sum()),
            "_per_seed_neg_group": neg_g.tolist()})
    return out


def check_totals(data, estimator, prof):
    """Οι ομάδες καλύπτουν όλους τους γύρους, άρα το άθροισμά τους πρέπει να δίνει το φ."""
    tot = np.sum([np.asarray(b["per_client_sum_mean"]) for b in prof], axis=0)
    ref = np.asarray(data["summary"]["per_estimator"][estimator]["phi_mean"])
    err = float(np.max(np.abs(tot - ref)))
    assert err < 1e-9, f"{estimator}: το άθροισμα των ομάδων διαφέρει από το φ κατά {err}"
    return err


def verdicts(res, neg):
    ck_k, ck_g = res["kernelshap"]["checkpoints"], res["gtg"]["checkpoints"]
    q_k, q_g = res["kernelshap"]["quarters"], res["gtg"]["quarters"]
    n_seeds = res["_n_seeds"]
    c15 = ck_g[0]["per_client_sum_mean"]
    return {
        "P1": {"gtg_1_10": ck_g[0]["neg_group_sum_mean"], "gtg_51_100": ck_g[3]["neg_group_sum_mean"],
               "pass": ck_g[0]["neg_group_sum_mean"] >= 0 and ck_g[3]["neg_group_sum_mean"] < 0},
        "P2": {"seeds_below_zero_51_100": ck_k[3]["seeds_neg_group_below_zero"], "of": n_seeds,
               "pass": ck_k[3]["seeds_neg_group_below_zero"] >= 15},
        "P3": {"seeds_at_or_above_zero_1_10": ck_k[0]["seeds_neg_group_at_or_above_zero"], "of": n_seeds,
               "pass": ck_k[0]["seeds_neg_group_at_or_above_zero"] >= 12},
        "P4": {"seeds_gross_ge_10x_net_51_100": ck_k[3]["seeds_gross_ge_10x_net"], "of": n_seeds,
               "pass": ck_k[3]["seeds_gross_ge_10x_net"] >= 15},
        "P5": {"kernelshap_q1_q4": [q_k[0]["neg_group_sum_mean"], q_k[3]["neg_group_sum_mean"]],
               "gtg_q1_q4": [q_g[0]["neg_group_sum_mean"], q_g[3]["neg_group_sum_mean"]],
               "pass": q_k[3]["neg_group_sum_mean"] < q_k[0]["neg_group_sum_mean"]
                       and q_g[3]["neg_group_sum_mean"] < q_g[0]["neg_group_sum_mean"]},
        "P6": {"gtg_1_10_most_negative_of_the_9": int(min(neg, key=lambda i: c15[i])),
               "pass": min(neg, key=lambda i: c15[i]) == 15}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Το φ ανά ομάδα γύρων.")
    p.add_argument("--src", default="results/contribution/har_contribution_ann_20seeds.json")
    p.add_argument("--out", default="results/contribution/har_contribution_temporal.json")
    add_overwrite_flag(p)
    args = p.parse_args(argv)
    check_output(args.out, args.overwrite)

    data = json.load(open(args.src, encoding="utf-8"))
    phi = np.asarray(data["summary"]["per_estimator"]["kernelshap"]["phi_mean"])
    neg = [int(i) for i in np.where(phi < 0)[0]]      # αρνητικό φ κατά KernelSHAP, μέσος των seeds

    res, checks = {"_n_seeds": len(data["per_seed"])}, {}
    for est in ESTIMATORS:
        res[est] = {sch: profile(data, est, sch, neg) for sch in SCHEMES}
        checks[est] = {sch: check_totals(data, est, res[est][sch]) for sch in SCHEMES}
    ver = verdicts(res, neg)
    for est in ESTIMATORS:
        for sch in SCHEMES:
            print(f"\n{est}, {sch}")
            for b in res[est][sch]:
                print(f"  γύροι {b['rounds'][0]:>3}-{b['rounds'][1]:<3}: αρνητικοί {b['neg_group_sum_mean']:+.3f}"
                      f" ± {b['neg_group_sum_std']:.3f} ({b['seeds_neg_group_below_zero']} seeds κάτω από 0),"
                      f" θετικοί {b['pos_group_sum_mean']:+.3f}")
    for k, v in ver.items():
        print(f"  {k}: {'επιβεβαιώθηκε' if v['pass'] else 'δεν επιβεβαιώθηκε'}")

    out = {"experiment": "contribution_temporal_profile", "source": args.src, "negative_set": neg,
           "negative_set_rule": "φ < 0 κατά KernelSHAP, μέσος των seeds (ίδιο σύνολο και για τον GTG-Tib)",
           "schemes": {k: [list(b) for b in v] for k, v in SCHEMES.items()},
           "total_check_max_abs_diff": checks,
           "note": "οι γύροι που παραλείφθηκαν δεν έχουν τιμές ανά γύρο και μετρούν ως 0",
           "results": res, "predictions": ver}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
