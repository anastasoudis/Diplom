# Σε ποιες δραστηριότητες χάνουν οι clients με αρνητικό φ.
#
# Ο πίνακας του contribution_classwise.py δίνει φ_i(R_c), τη συνεισφορά του client i
# στο recall R_c της δραστηριότητας c. Αφού φ_i = Σ_c w_c φ_i(R_c), με w_c το ποσοστό
# της c στο test, το φ κάθε client χωρίζεται σε έξι όρους w_c φ_i(R_c). Εδώ οι όροι
# αθροίζονται για την ομάδα με αρνητικό φ και για την ομάδα με θετικό.
#
#   python -m implementation.contribution.classwise_groups

import argparse
import json
import os
import sys

import numpy as np

from implementation.common.outputs import add_overwrite_flag, check_output

CW_FILE = "results/contribution/har_contribution_classwise.json"
PHI_FILE = "results/contribution/har_contribution_ann_20seeds.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Οι όροι w_c φ_i(R_c) ανά ομάδα clients.")
    ap.add_argument("--estimator", default="kernelshap_cw")
    ap.add_argument("--out", default="results/contribution/har_classwise_groups.json")
    add_overwrite_flag(ap)
    a = ap.parse_args(argv)
    check_output(a.out, a.overwrite)

    cw = json.load(open(CW_FILE, encoding="utf-8"))
    acts, w = cw["activities"], np.array(cw["class_weights"])
    M = np.array(cw["phi"][a.estimator])
    phi = {r["client"]: r["phi_kernelshap"]
           for r in json.load(open(PHI_FILE, encoding="utf-8"))["summary"]["clients"]}
    y = np.array([phi[i] for i in range(M.shape[0])])

    T = M * w
    # οι όροι κάθε client πρέπει να αθροίζουν στο φ του, μέσα στην ανοχή του ελέγχου
    # ταυτότητας που έκανε το contribution_classwise.py
    gap = float(np.max(np.abs(T.sum(1) - y)))
    tol = cw["identity_check"][a.estimator]["max_abs_diff"] * 1.01
    if gap > tol:
        raise SystemExit(f"οι όροι δεν δίνουν το φ: μέγιστη διαφορά {gap:.2e}, ανοχή {tol:.2e}")
    print(f"  μέγιστη διαφορά Σ_c w_c φ_i(R_c) και φ_i: {gap:.2e}")

    neg = y < 0
    out = {"experiment": "classwise_groups", "classwise_source": CW_FILE, "phi_source": PHI_FILE,
           "estimator": a.estimator, "identity_max_abs_diff": gap, "activities": acts,
           "class_weights": w.tolist()}
    for name, mask in (("negative", neg), ("positive", ~neg)):
        s = T[mask].sum(0)
        out[name] = {"clients": [int(i) for i in np.where(mask)[0]],
                     "by_activity": {c: float(v) for c, v in zip(acts, s)}, "total": float(s.sum())}
        print(f"  {name:<9} ({mask.sum():>2}): " + ", ".join(f"{c} {v:+.2f}" for c, v in zip(acts, s))
              + f", σύνολο {s.sum():+.2f}")
    sit, st = acts.index("SITTING"), acts.index("STANDING")
    out["negative_with_sitting_neg_standing_pos"] = [
        int(i) for i in np.where(neg & (M[:, sit] < 0) & (M[:, st] > 0))[0]]
    out["per_client_terms"] = {int(i): {c: float(v) for c, v in zip(acts, T[i])} for i in range(len(y))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
