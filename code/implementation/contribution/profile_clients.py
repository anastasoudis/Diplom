# Περιγραφή των clients και σύγκριση όσων έχουν αρνητικό φ με τους υπόλοιπους.
#
# Για κάθε client υπολογίζονται, χωρίς να εκπαιδευτεί τίποτα, το πλήθος των samples, η
# απόσταση των δεδομένων του από το test και από τους άλλους clients και η διασπορά μέσα
# σε κάθε δραστηριότητα. Μαζί τους η επίδοση του μοντέλου του στην ατομική εκπαίδευση
# (σκέλος Α).
#
# Η απόσταση υπολογίζεται ανά δραστηριότητα και σταθμίζεται με τη σύνθεση του test:
#   d(p, q) = Σ_c w_c ||μ_c(p) - μ_c(q)||
# με μ_c τον μέσο των 561 χαρακτηριστικών στη δραστηριότητα c. Έτσι η απόσταση δεν
# επηρεάζεται από το πόσα samples έχει κάθε δραστηριότητα. Δραστηριότητες που λείπουν
# από κάποιον παραλείπονται και τα βάρη κανονικοποιούνται ξανά.
#
# Περιγραφική ανάλυση σε 21 clients. Καμία συσχέτιση δεν δείχνει αιτία. Η κατάταξη
# των χαρακτηριστικών δεν έχει διόρθωση πολλαπλών ελέγχων.
#
#   python -m implementation.contribution.profile_clients

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.dataset.export_processed import OUT_DIR, load_processed

ACTS = ["WALKING", "W_UPSTAIRS", "W_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
PHI_FILE = "results/contribution/har_contribution_ann_20seeds.json"
LOCAL_FILE = "results/local_only/har_local_ann.json"


def profile(X, y, n_classes=6):
    cnt = np.array([int((y == c).sum()) for c in range(n_classes)], dtype=float)
    frac = cnt / cnt.sum()
    nz = frac[frac > 0]
    mu_c = np.full((n_classes, X.shape[1]), np.nan)
    sd_c = np.full((n_classes, X.shape[1]), np.nan)
    for c in range(n_classes):
        m = y == c
        if m.sum() >= 2:
            mu_c[c] = X[m].mean(0)
            sd_c[c] = X[m].std(0, ddof=1)
    return {"n": int(len(y)), "counts": cnt, "frac": frac,
            "entropy": float(-(nz * np.log(nz)).sum()),
            "mu": X.mean(0), "mu_c": mu_c, "sd_c": sd_c,
            "spread": float(np.nanmean(sd_c))}      # μέση διασπορά μέσα στις δραστηριότητες


def cc_distance(p, q, weights):
    d, w = [], []
    for c in range(len(weights)):
        a, b = p["mu_c"][c], q["mu_c"][c]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        d.append(float(np.linalg.norm(a - b)))
        w.append(weights[c])
    if not d:
        return float("nan")
    return float(np.dot(np.array(w) / np.sum(w), d))


def kl(p, q, eps=1e-9):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Περιγραφή των clients.")
    ap.add_argument("--out", default="results/contribution/har_client_profiles.json")
    add_overwrite_flag(ap)
    a = ap.parse_args(argv)
    check_output(a.out, a.overwrite)

    trainset, testset, cid, prov = load_processed()
    Xtr, ytr = trainset.X.numpy().reshape(len(trainset), -1), trainset.y.numpy()
    Xte, yte = testset.X.numpy().reshape(len(testset), -1), testset.y.numpy()
    te_sid = testset.subject_ids
    orig = prov["meta"]["subject_ids_original"]     # client 0-20 και ο αριθμός του στο UCI
    test_ids = prov["meta"]["test_subjects"]
    with open(os.path.join(OUT_DIR, "feature_names.csv"), encoding="utf-8") as fh:
        names = [row["name"] for row in csv.DictReader(fh)]

    cl = {i: profile(Xtr[cid == i], ytr[cid == i]) for i in range(int(cid.max()) + 1)}
    te = {s: profile(Xte[te_sid == s], yte[te_sid == s]) for s in test_ids}
    pooled_test = profile(Xte, yte)
    w = pooled_test["frac"]

    phi = {r["client"]: r["phi_kernelshap"]
           for r in json.load(open(PHI_FILE, encoding="utf-8"))["summary"]["clients"]}
    a_acc = {}
    for ps in json.load(open(LOCAL_FILE, encoding="utf-8"))["results"]["subject"]["per_seed"]:
        for c in ps["clients"]:
            a_acc.setdefault(c["client"], []).append(c["A_global_acc"])
    a_acc = {k: float(np.mean(v)) for k, v in a_acc.items()}

    ids = sorted(cl)
    neg = [i for i in ids if phi[i] < 0]
    pos = [i for i in ids if phi[i] >= 0]
    V = {"n_samples": np.array([cl[i]["n"] for i in ids], float),
         "d_test": np.array([cc_distance(cl[i], pooled_test, w) for i in ids]),
         # απόσταση από τους άλλους clients, για να ξεχωρίζει το «διαφορετικός από όλους»
         # από το «διαφορετικός από το test»
         "d_train": np.array([float(np.mean([cc_distance(cl[i], cl[j], w) for j in ids if j != i]))
                              for i in ids])}
    V["d_test_minus_train"] = V["d_test"] - V["d_train"]
    V["label_KL_to_test"] = np.array([kl(cl[i]["frac"], w) for i in ids])
    V["label_entropy"] = np.array([cl[i]["entropy"] for i in ids])
    V["spread"] = np.array([cl[i]["spread"] for i in ids])
    V["individualA_acc"] = np.array([a_acc[i] for i in ids])
    y_phi = np.array([phi[i] for i in ids])

    # για σύγκριση, η μέση απόσταση κάθε ατόμου του test από τα άλλα 8
    te_d = np.array([float(np.mean([cc_distance(te[s], te[t], w) for t in test_ids if t != s]))
                     for s in test_ids])

    print(f"clients με αρνητικό φ: {neg}, στο UCI {[orig[i] for i in neg]}")
    idx = {i: k for k, i in enumerate(ids)}
    print(f"\n{'μέγεθος':<20}{'αρνητικοί':>12}{'θετικοί':>12}{'Mann-Whitney p':>16}"
          f"{'Pearson r με φ':>16}{'Spearman ρ':>12}")
    for k, v in V.items():
        g1, g2 = v[[idx[i] for i in neg]], v[[idx[i] for i in pos]]
        print(f"{k:<20}{g1.mean():>12.4f}{g2.mean():>12.4f}"
              f"{mannwhitneyu(g1, g2, alternative='two-sided').pvalue:>16.4f}"
              f"{pearsonr(v, y_phi)[0]:>16.4f}{spearmanr(v, y_phi)[0]:>12.4f}")
    print(f"\nμέση απόσταση ατόμου του test από τα άλλα 8: {te_d.mean():.3f}, "
          f"μέση απόσταση client από το test: {V['d_test'].mean():.3f}")

    # Ποια χαρακτηριστικά ξεχωρίζουν τις δύο ομάδες, με μονάδα τον client (όχι το sample).
    # Cohen d = διαφορά μέσων / συγκεντρωτική τυπική απόκλιση.
    Mn, Mp = np.array([cl[i]["mu"] for i in neg]), np.array([cl[i]["mu"] for i in pos])
    pooled = np.sqrt((Mn.var(0, ddof=1) * (len(neg) - 1) + Mp.var(0, ddof=1) * (len(pos) - 1))
                     / (len(neg) + len(pos) - 2))
    d_cohen = (Mn.mean(0) - Mp.mean(0)) / np.where(pooled > 1e-12, pooled, np.nan)
    order = np.argsort(-np.abs(np.nan_to_num(d_cohen)))
    print("\nτα 10 χαρακτηριστικά με το μεγαλύτερο |d|:")
    for f in order[:10]:
        print(f"  {f + 1:>4}  d = {d_cohen[f]:+.3f}  {names[f]}")
    fam, tot = Counter(), Counter()
    for f in range(len(names)):
        key = names[f].split("-")[0]
        tot[key] += 1
        fam[key] += int(abs(np.nan_to_num(d_cohen[f])) > 0.8)

    out = {"clients": [{"client": int(i), "subject_id_original": int(orig[i]), "n_samples": cl[i]["n"],
                        "phi_kernelshap": phi[i], "individualA_acc": a_acc[i],
                        **{k: float(V[k][idx[i]]) for k in V},
                        "class_fraction": {ACTS[c]: float(cl[i]["frac"][c]) for c in range(6)}}
                       for i in ids],
           "phi_source": PHI_FILE,
           "negative_clients": [int(i) for i in neg],
           "negative_subjects_original": [int(orig[i]) for i in neg],
           "test_subjects": [int(s) for s in test_ids],
           "test_internal_distance": {"mean": float(te_d.mean()), "min": float(te_d.min()),
                                      "max": float(te_d.max())},
           "top_features": [{"index": int(f + 1), "name": names[f], "cohen_d": float(d_cohen[f])}
                            for f in order[:40]],
           "feature_families_abs_d_gt_0_8": {k: [fam[k], tot[k]] for k in tot if fam[k]},
           "notes": ["Περιγραφική ανάλυση, καμία συσχέτιση δεν δείχνει αιτία.",
                     "21 clients, οπότε τα p είναι ενδεικτικά.",
                     "Η κατάταξη των χαρακτηριστικών δεν έχει διόρθωση πολλαπλών ελέγχων."]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
