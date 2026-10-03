# Σε ποια άτομα του test βλάπτουν οι clients με αρνητικό φ.
#
# Διαβάζει τον πίνακα φ 21 x 9 του contribution_classwise.py --value_metric per_subject,
# όπου φ_{i,s} είναι η συνεισφορά του client i στο accuracy του ατόμου s. Με w_s το
# ποσοστό των samples του s στο test, ισχύει Σ_s w_s φ_{i,s} = φ_i. Δεν εκπαιδεύει
# τίποτα. Οι προβλέψεις P7-P11 γράφτηκαν πριν από την ανάλυση.
#
#   python -m implementation.contribution.per_subject_analysis

import argparse
import json
import os
import sys

import numpy as np

from implementation.common.outputs import add_overwrite_flag, check_output

HEADLINE_PHI = "results/contribution/har_contribution_ann_20seeds.json"


def main(argv=None):
    p = argparse.ArgumentParser(description="Το φ ανά άτομο του test για τους αρνητικούς clients.")
    p.add_argument("--src", default="results/contribution/har_contribution_per_subject.json")
    p.add_argument("--out", default="results/contribution/har_contribution_per_subject_analysis.json")
    add_overwrite_flag(p)
    a = p.parse_args(argv)
    check_output(a.out, a.overwrite)

    d = json.load(open(a.src, encoding="utf-8"))
    labels = d["per_seed"][0]["output_labels"]
    w = np.asarray(d["class_weights"])
    phi = np.asarray(d["phi"]["kernelshap_cw"])
    head = np.asarray(json.load(open(HEADLINE_PHI))["summary"]["per_estimator"]["kernelshap"]["phi_mean"])
    neg = [int(i) for i in np.where(head < 0)[0]]

    weighted = phi * w                      # οι όροι w_s φ_{i,s}, που αθροίζουν στο φ_i
    col = weighted[neg].sum(0)              # οι αρνητικοί μαζί, ανά άτομο του test
    neg_cols = col[col < 0]
    share = float(col.min() / neg_cols.sum()) if len(neg_cols) else 0.0
    col_std = phi.std(0, ddof=1)
    ic = d["identity_check"]
    res = {"labels": labels, "weights": w.tolist(), "negatives": neg,
           "phi_mean_kernelshap": phi.tolist(),
           "neg_group_weighted_by_subject": dict(zip(map(str, labels), col.tolist())),
           "most_harmed_subject": int(labels[int(np.argmin(col))]),
           "per_negative": {str(i): {"n_subjects_positive": int((phi[i] > 0).sum()),
                                     "n_subjects_negative": int((phi[i] < 0).sum()),
                                     "worst_subject": int(labels[int(np.argmin(phi[i]))]),
                                     "best_subject": int(labels[int(np.argmax(phi[i]))])} for i in neg},
           "column_std_across_clients": dict(zip(map(str, labels), col_std.tolist()))}
    helping = int(sum((phi[i] > 0).any() for i in neg))
    res["predictions"] = {
        "P7": {"kernelshap": ic["kernelshap_cw"].get("max_abs_diff"), "gtg": ic["gtg_cw"].get("max_abs_diff"),
               "pass": ic["kernelshap_cw"].get("max_abs_diff", 1) < 0.01 and ic["gtg_cw"].get("max_abs_diff", 1) < 0.05},
        "P8": {"negatives_helping_someone": helping, "pass": helping >= 6},
        "P9": {"client2_subjects_negative": int((phi[2] < 0).sum()), "pass": int((phi[2] < 0).sum()) >= 7},
        "P10": {"share_of_most_negative_column": share, "pass": share >= 0.25},
        "P11": {"max_over_min_column_std": float(col_std.max() / col_std.min()),
                "pass": float(col_std.max() / col_std.min()) >= 2}}

    print("φ ανά άτομο του test για τους αρνητικούς clients (KernelSHAP)")
    print("  client " + "".join(f"{'S' + str(s):>9}" for s in labels) + "   Σ w φ")
    for i in neg:
        print(f"  {i:>6} " + "".join(f"{phi[i, k]:>9.3f}" for k in range(len(labels))) + f"   {weighted[i].sum():+.3f}")
    for k, x in res["predictions"].items():
        print(f"  {k}: {'επιβεβαιώθηκε' if x['pass'] else 'δεν επιβεβαιώθηκε'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"experiment": "per_subject_analysis", "source": a.src, **res}, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
