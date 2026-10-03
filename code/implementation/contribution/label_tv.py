# Σχετίζεται η σύνθεση δραστηριοτήτων κάθε client με το φ του;
#
# Για κάθε client η απόσταση ολικής μεταβολής (TV) της κατανομής των δραστηριοτήτων
# του από την κατανομή όλου του train και από την κατανομή του test:
#   TV(p, q) = Σ_c |p_c - q_c| / 2, από 0 (ίδιες κατανομές) έως 1
# Ο υπολογισμός είναι ίδιος με της διερευνητικής ανάλυσης (eda_uci.py), οπότε η διάμεσος
# των 21 τιμών ελέγχεται απέναντι σε εκείνη. Περιγραφική συσχέτιση σε 21 clients.
#
#   python -m implementation.contribution.label_tv

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.dataset.export_processed import load_processed

PHI_FILE = "results/contribution/har_contribution_ann_20seeds.json"
EDA_FILE = "results/dataset/eda_uci.json"


def fractions(y, n_classes=6):
    return np.array([np.mean(y == c) for c in range(n_classes)])


def tv(p, q):
    return float(np.abs(np.asarray(p) - np.asarray(q)).sum() / 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Απόσταση κατανομής δραστηριοτήτων ανά client και φ.")
    ap.add_argument("--out", default="results/contribution/har_label_tv.json")
    add_overwrite_flag(ap)
    a = ap.parse_args(argv)
    check_output(a.out, a.overwrite)

    trainset, testset, cid, _ = load_processed()
    ytr, yte = trainset.y.numpy(), testset.y.numpy()
    ids = list(range(int(cid.max()) + 1))
    glob, test = fractions(ytr), fractions(yte)
    tv_glob = np.array([tv(fractions(ytr[cid == i]), glob) for i in ids])
    tv_test = np.array([tv(fractions(ytr[cid == i]), test) for i in ids])
    med = float(np.median(tv_glob))
    if os.path.exists(EDA_FILE):
        ref = json.load(open(EDA_FILE, encoding="utf-8"))["clients"]["tv_median"]
        if abs(med - ref) > 1e-9:
            raise SystemExit(f"η διάμεσος {med} διαφέρει από του {EDA_FILE} ({ref})")

    phi = {r["client"]: r["phi_kernelshap"]
           for r in json.load(open(PHI_FILE, encoding="utf-8"))["summary"]["clients"]}
    y = np.array([phi[i] for i in ids])
    out = {"experiment": "label_tv_vs_phi", "phi_source": PHI_FILE, "reference_median_source": EDA_FILE,
           "tv_median": med, "tv_max": float(tv_glob.max()),
           "clients": [{"client": i, "phi_kernelshap": float(phi[i]),
                        "tv_to_global": float(tv_glob[k]), "tv_to_test": float(tv_test[k])}
                       for k, i in enumerate(ids)]}
    for name, v in (("tv_to_global", tv_glob), ("tv_to_test", tv_test)):
        r, p = pearsonr(v, y)
        rho, ps = spearmanr(v, y)
        z = (v - v.mean()) / v.std()
        out[name] = {"pearson_r": float(r), "p": float(p), "spearman_rho": float(rho),
                     "spearman_p": float(ps), "z_client2": float(z[2])}
        print(f"  φ και {name}: r {r:+.4f} (p {p:.4f}), ρ {rho:+.4f} (p {ps:.4f}), z του client 2 {z[2]:+.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
