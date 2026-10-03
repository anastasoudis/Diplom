# Η κλίμακα των δεδομένων πριν και μετά τη z-score (Κεφάλαιο 3, πίνακας κλίμακας).
#
# Για train και test δίνει μέσο, τυπική απόκλιση και εύρος (α) στην κλίμακα της
# πηγής, (β) μετά τη z-score με στατιστικά μόνο του train, όπως τα δεδομένα που
# βλέπει το μοντέλο και (γ) μετά από z-score με στατιστικά της ένωσης train και
# test, δηλαδή με το λάθος που κάνει η min-max της πηγής.
#
#   python -m implementation.dataset.scaling_check

import json
import os
import sys

import numpy as np
import pandas as pd

from implementation.dataset.load_dataset import ROOT, UCI_DIR

PROC = os.path.join(ROOT, "data", "har_processed")
OUT = os.path.join(ROOT, "results", "dataset", "scaling_check_uci.json")


def summary(A):
    # ο μέσος όλων των τιμών και η μέση τυπική απόκλιση των 561 στηλών (ddof=0,
    # όπως στη z-score)
    return {"mean": float(A.mean()),
            "col_std_mean": float(A.std(axis=0).mean()),
            "min": float(A.min()), "max": float(A.max())}


def main():
    raw_tr = np.loadtxt(os.path.join(UCI_DIR, "train", "X_train.txt"))
    raw_te = np.loadtxt(os.path.join(UCI_DIR, "test", "X_test.txt"))
    z_tr = pd.read_csv(os.path.join(PROC, "X_train.csv"), header=None).to_numpy(np.float64)
    z_te = pd.read_csv(os.path.join(PROC, "X_test.csv"), header=None).to_numpy(np.float64)

    U = np.vstack([raw_tr, raw_te])
    mu_u, sd_u = U.mean(axis=0), U.std(axis=0)

    # Έλεγχος ότι τα επεξεργασμένα αρχεία είναι πράγματι z-score με στατιστικά του
    # train. Η διαφορά που μένει οφείλεται στο float32 και στις 6 δεκαδικές θέσεις.
    mu, sd = raw_tr.mean(axis=0), raw_tr.std(axis=0)
    diff = max(np.abs((raw_tr - mu) / sd - z_tr).max(), np.abs((raw_te - mu) / sd - z_te).max())

    rep = {"what": "κλίμακα train και test πριν και μετά τη z-score",
           "command": "python -m implementation.dataset.scaling_check",
           "source": {"train": summary(raw_tr), "test": summary(raw_te)},
           "zscore_train_stats": {"train": summary(z_tr), "test": summary(z_te)},
           "zscore_union_stats_counterfactual": {"train": summary((raw_tr - mu_u) / sd_u),
                                                 "test": summary((raw_te - mu_u) / sd_u)},
           "max_abs_diff_processed_vs_recomputed": float(diff)}
    for k in ("source", "zscore_train_stats", "zscore_union_stats_counterfactual"):
        for s in ("train", "test"):
            r = rep[k][s]
            print(f"{k:34s} {s:5s} μέσος {r['mean']:+.5f}  τ.α. {r['col_std_mean']:.5f}"
                  f"  εύρος [{r['min']:+.2f}, {r['max']:+.2f}]")
    print(f"επεξεργασμένα αρχεία έναντι επανυπολογισμού: {diff:.1e}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"γράφτηκε το {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
