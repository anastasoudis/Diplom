# Διερευνητική ανάλυση του UCI HAR (Κεφάλαιο 3 της εργασίας).
#
# Οι ενότητες για τα κανάλια, τη χρονική ανάλυση, τις συχνότητες και τις
# συσχετίσεις καναλιών υπολογίζονται από τα ωμά σήματα 9x128. Είναι περιγραφικές,
# γιατί το μοντέλο εκπαιδεύεται στα 561 χαρακτηριστικά και όχι στα σήματα.
# Όπου έχει σημασία, γράφεται αν ένα μέγεθος αφορά τα ωμά δεδομένα (RAW) ή τα
# δεδομένα μετά τη z-score (MODEL).
#
#   python -m implementation.dataset.eda_uci

import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import kurtosis, shapiro, skew
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from implementation.dataset.load_dataset import (ACTIVITIES, ROOT, UCI_CHANNELS, UCI_DIR,
                                                 load_uci_har, load_uci_signals)

OUT = os.path.join(ROOT, "results", "dataset", "eda_uci.json")
SEED = 42
FS = 50.0          # Hz
# Μία γεννήτρια για όλη την ανάλυση. Οι ενότητες καλούνται πάντα με την ίδια
# σειρά, οπότε τα δείγματα που παίρνει η καθεμία είναι πάντα τα ίδια.
rng = np.random.default_rng(SEED)


def title(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _j(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))


def sec1_integrity(Xr, Xm, y, s, S, rep):
    title("3.2.1 Ακεραιότητα και ποιότητα")
    print(f"  samples {len(Xr)}, χαρακτηριστικά {Xr.shape[1]}, σήματα {S.shape}")
    print(f"  NaN: X {int(np.isnan(Xr).sum())}, σήματα {int(np.isnan(S).sum())}")
    print(f"  Inf: X {int(np.isinf(Xr).sum())}, σήματα {int(np.isinf(S).sum())}")
    const = int((Xr.std(0) == 0).sum())
    print(f"  σταθερές στήλες: {const}/561")
    # Δύο στήλες μπορεί να έχουν κανονική διακύμανση και να είναι ίδιες byte προς
    # byte, κάτι που ο έλεγχος std == 0 δεν πιάνει.
    first, groups = {}, {}
    for j in range(Xr.shape[1]):
        k = Xr[:, j].tobytes()
        if k in first:
            groups.setdefault(first[k], []).append(j)
        else:
            first[k] = j
    n_red = sum(len(v) for v in groups.values())
    print(f"  ίδιες στήλες: {len(groups)} ομάδες, {n_red} πλεονάζουσες")
    outside = int(((Xr.min(0) < -1.0001) | (Xr.max(0) > 1.0001)).sum())
    print(f"  εύρος RAW [{Xr.min():+.4f}, {Xr.max():+.4f}], στήλες εκτός [-1, 1]: {outside}/561")
    print(f"  εύρος MODEL [{Xm.min():+.4f}, {Xm.max():+.4f}]")
    rep["integrity"] = {"n_windows": len(Xr), "n_features": Xr.shape[1],
                        "signals_shape": list(S.shape), "nan_X": int(np.isnan(Xr).sum()),
                        "nan_signals": int(np.isnan(S).sum()), "constant_columns": const,
                        "raw_range": [float(Xr.min()), float(Xr.max())],
                        "model_range": [float(Xm.min()), float(Xm.max())],
                        "cols_outside_pm1": outside,
                        "identical_column_groups": len(groups),
                        "redundant_columns": n_red,
                        "identical_columns": {str(a + 1): [x + 1 for x in v]
                                              for a, v in sorted(groups.items())}}


def sec2_classes(y_tr, y_te, rep):
    title("3.2.2 Κατανομή κλάσεων")
    out = {}
    for nm, y in (("train", y_tr), ("test", y_te)):
        c = Counter(y.tolist())
        n = len(y)
        print(f"  {nm} (n={n})")
        for k in sorted(c):
            print(f"    {ACTIVITIES[k]:<20s} {c[k]:>5d}  {100 * c[k] / n:5.2f}%")
        p = np.array([c[k] / n for k in sorted(c)])
        imb = max(c.values()) / min(c.values())
        # εντροπία διαιρεμένη με log(6), ώστε το 1 να σημαίνει απόλυτα ισορροπημένες κλάσεις
        H = float(-(p * np.log(p)).sum() / np.log(len(p)))
        print(f"    λόγος ανισορροπίας {imb:.3f}, κανονικοποιημένη εντροπία {H:.4f}")
        out[nm] = {"counts": {ACTIVITIES[k]: int(v) for k, v in sorted(c.items())},
                   "imbalance_ratio": float(imb), "normalized_entropy": H}
    rep["classes"] = out


def sec3_per_channel(S, y, rep):
    title("3.2.3 Περιγραφικά στατιστικά ανά κανάλι (ωμά σήματα)")
    out = {}
    for i, c in enumerate(UCI_CHANNELS):
        v = S[:, i, :].ravel()
        print(f"  {c:<14s} μέσος {v.mean():8.4f}  τ.α. {v.std():7.4f}  "
              f"εύρος [{v.min():.3f}, {v.max():.3f}]")
        out[c] = {"mean": float(v.mean()), "std": float(v.std()),
                  "min": float(v.min()), "max": float(v.max())}
    print("  τυπική απόκλιση του body_acc_x ανά δραστηριότητα")
    for k in sorted(set(y.tolist())):
        v = S[y == k, 0, :]
        print(f"    {ACTIVITIES[k]:<20s} {v.std():.4f}")
        out.setdefault("body_acc_x_std_per_activity", {})[ACTIVITIES[k]] = float(v.std())
    rep["per_channel"] = out


def sec4_normality(Xm, rep):
    title("3.2.4 Κατανομή τιμών και κανονικότητα")
    n, d = Xm.shape
    sub = rng.choice(n, size=min(n, 4000), replace=False)
    p_real = np.array([shapiro(Xm[sub, j]).pvalue for j in range(d)])
    # Ο ίδιος έλεγχος σε δείγματα από πραγματικά κανονική κατανομή δείχνει πόσο
    # συχνά απορρίπτει όταν δεν θα έπρεπε (περίπου 5%).
    ref = rng.standard_normal((len(sub), 100))
    p_ref = np.array([shapiro(ref[:, j]).pvalue for j in range(100)])
    rej, rej_ref = float((p_real < .05).mean()), float((p_ref < .05).mean())
    print(f"  Shapiro-Wilk σε {len(sub)} samples: απορρίπτει σε {100 * rej:.1f}% των {d} "
          f"χαρακτηριστικών, σε {100 * rej_ref:.1f}% των κανονικών δειγμάτων")
    sk, ku = skew(Xm, axis=0), kurtosis(Xm, axis=0)
    near = float(((abs(sk) < .5) & (abs(ku) < 1.)).mean())
    heavy = float((ku > 3.).mean())
    print(f"  διάμεσος |ασυμμετρίας| {np.median(abs(sk)):.2f}, "
          f"διάμεσος |περίσσειας κύρτωσης| {np.median(abs(ku)):.2f}")
    print(f"  κοντά στην κανονική {100 * near:.1f}%, βαριά ουρά {100 * heavy:.1f}%")
    rep["normality"] = {"shapiro_reject_rate": rej, "shapiro_reject_on_true_normal": rej_ref,
                        "abs_skew_median": float(np.median(abs(sk))),
                        "abs_kurtosis_median": float(np.median(abs(ku))),
                        "frac_near_normal": near, "frac_heavy_tailed": heavy}


def sec5_temporal(S, y, rep):
    title("3.2.5 Χρονική ανάλυση (ωμά σήματα)")
    # Αυτοσυσχέτιση του body_acc_x μέσα στο sample. Η πρώτη κορυφή μετά το lag 5
    # δίνει την περίοδο του βήματος.
    out = {}
    for k in sorted(set(y.tolist())):
        w = S[y == k, 0, :]
        w = w - w.mean(1, keepdims=True)
        acf = np.array([[np.dot(r[:128 - L], r[L:]) / (np.dot(r, r) + 1e-12)
                         for L in range(64)] for r in w[:400]]).mean(0)
        lag = int(np.argmax(acf[5:]) + 5)
        print(f"  {ACTIVITIES[k]:<20s} lag {lag:>3d}, περίοδος {lag / FS:.3f} s, ACF {acf[lag]:.4f}")
        out[ACTIVITIES[k]] = {"peak_lag": lag, "period_s": lag / FS, "acf_at_peak": float(acf[lag])}
    rep["temporal"] = out


def sec6_frequency(S, y, rep):
    title("3.2.6 Ανάλυση συχνοτήτων (ωμά σήματα)")
    freqs = np.fft.rfftfreq(128, d=1 / FS)
    out = {}
    for k in sorted(set(y.tolist())):
        w = S[y == k, 0, :]
        w = w - w.mean(1, keepdims=True)
        P = (np.abs(np.fft.rfft(w, axis=1)) ** 2).mean(0)
        P[0] = 0
        dom = float(freqs[int(np.argmax(P))])
        lo = float(P[(freqs > 0) & (freqs < 3)].sum() / P.sum())
        hi = float(P[freqs > 5].sum() / P.sum())
        print(f"  {ACTIVITIES[k]:<20s} κυρίαρχη {dom:5.2f} Hz, ισχύς κάτω από 3 Hz {100 * lo:5.1f}%, "
              f"πάνω από 5 Hz {100 * hi:5.1f}%")
        out[ACTIVITIES[k]] = {"dominant_hz": dom, "power_below_3hz": lo, "power_above_5hz": hi}
    rep["frequency"] = out


def sec7_channel_corr(S, rep):
    title("3.2.7 Συσχετίσεις καναλιών (ωμά σήματα)")
    M = S.transpose(1, 0, 2).reshape(9, -1)
    C = np.corrcoef(M)
    off = C[~np.eye(9, dtype=bool)]
    print(f"  μέσο |r| εκτός διαγωνίου {np.abs(off).mean():.4f}, μέγιστο {np.abs(off).max():.4f}")
    rep["channel_corr"] = {"matrix": C.tolist(), "mean_abs_offdiag": float(np.abs(off).mean()),
                           "max_abs_offdiag": float(np.abs(off).max())}


def sec8_separability(Xm, y, rep):
    title("3.2.8 Διαχωρισιμότητα κλάσεων")
    idx = rng.choice(len(Xm), size=min(4000, len(Xm)), replace=False)
    Z = PCA(n_components=2, random_state=SEED).fit_transform(Xm[idx])
    sil2 = float(silhouette_score(Z, y[idx]))
    silf = float(silhouette_score(Xm[idx], y[idx]))
    pca = PCA(n_components=50, random_state=SEED).fit(Xm[idx])
    ev = float(pca.explained_variance_ratio_[:2].sum())
    # οι τρεις δραστηριότητες βαδίσματος απέναντι στις τρεις στατικές
    grp = np.where(np.isin(y[idx], [1, 2, 3]), 0, 1)
    sil_mv = float(silhouette_score(Xm[idx], grp))
    print(f"  silhouette στις 2 πρώτες συνιστώσες PCA {sil2:+.4f} ({100 * ev:.1f}% της διακύμανσης)")
    print(f"  silhouette στις 561 διαστάσεις {silf:+.4f}, κίνηση και ακινησία {sil_mv:+.4f}")
    rep["separability"] = {"silhouette_pca2": sil2, "silhouette_full": silf,
                           "pca2_explained": ev, "silhouette_moving_vs_static": sil_mv}


def sec9_outliers(Xm, y, sid, Xm_te, rep):
    title("3.2.9 Ακραίες τιμές σε σύγκριση με κανονικά δεδομένα")
    # Ένας ανιχνευτής όπως το IsolationForest με contamination 0,05 επιστρέφει 5%
    # εξ ορισμού. Εδώ η σύγκριση γίνεται με το πόσα ακραία θα είχαν δεδομένα ίδιου
    # μεγέθους από κανονική κατανομή.
    n, d = Xm.shape
    obs_val = float((np.abs(Xm) > 3).mean())
    obs_row = float((np.abs(Xm) > 3).any(1).mean())
    ref = rng.standard_normal((n, d))
    exp_val = float((np.abs(ref) > 3).mean())
    exp_row = float((np.abs(ref) > 3).any(1).mean())
    print(f"  τιμές με |z| > 3: {100 * obs_val:.2f}% (κανονικά δεδομένα {100 * exp_val:.2f}%)")
    print(f"  samples με τουλάχιστον μία: {100 * obs_row:.2f}% (κανονικά δεδομένα {100 * exp_row:.2f}%)")
    cnt = (np.abs(Xm) > 3).sum(1)
    cnt_ref = (np.abs(ref) > 3).sum(1)
    C = np.corrcoef(Xm[rng.choice(n, min(2000, n), replace=False)].T)
    mabs = float(np.abs(C[~np.eye(d, dtype=bool)]).mean())
    print(f"  μέγιστο πλήθος ακραίων σε ένα sample {int(cnt.max())} (κανονικά δεδομένα {int(cnt_ref.max())})")
    print(f"  μέσο |r| μεταξύ χαρακτηριστικών {mabs:.4f}")

    # Πού μαζεύονται τα samples με πολλές ακραίες τιμές. Αν δένονται με
    # δραστηριότητα ή άτομο, είναι σήμα και όχι σφάλμα μέτρησης. Το όριο 21 είναι
    # πάνω από το μέγιστο που δίνουν τα κανονικά δεδομένα.
    THR = 21
    heavy = cnt >= THR
    per_act = {}
    for k in sorted(set(y.tolist())):
        m = y == k
        per_act[ACTIVITIES[k]] = {"mean_extremes": float(cnt[m].mean()),
                                  "frac_heavy": float(heavy[m].mean()),
                                  "max_extremes": int(cnt[m].max())}
        print(f"    {ACTIVITIES[k]:<20s} μέσο πλήθος {cnt[m].mean():6.2f}, "
              f"με {THR} ή περισσότερες {100 * heavy[m].mean():5.2f}%")
    per_cl = []
    for c in sorted(set(sid.tolist())):
        m = sid == c
        per_cl.append((float(heavy[m].mean()), int(c), int(m.sum()), float(cnt[m].mean())))
    per_cl.sort(reverse=True)
    top3 = sum(f * sz for f, _, sz, _ in per_cl[:3]) / max(1, heavy.sum())
    print(f"  μερίδιο των 3 clients με τα περισσότερα: {100 * top3:.1f}%")

    # Αν το test είχε πολύ διαφορετικό ποσοστό, θα ήταν μετατόπιση κατανομής.
    cnt_te = (np.abs(Xm_te) > 3).sum(1)
    print(f"  test: τιμές με |z| > 3 {100 * float((np.abs(Xm_te) > 3).mean()):.2f}%, "
          f"samples με {THR} ή περισσότερες {100 * float((cnt_te >= THR).mean()):.2f}%")

    rep["outliers"] = {"observed_value_rate": obs_val, "null_value_rate": exp_val,
                       "observed_row_rate": obs_row, "null_row_rate": exp_row,
                       "max_extremes_in_row": int(cnt.max()),
                       "null_max_extremes_in_row": int(cnt_ref.max()),
                       "mean_abs_feature_corr": mabs,
                       "heavy_row_threshold": THR,
                       "heavy_rows_train": int(heavy.sum()),
                       "frac_heavy_train": float(heavy.mean()),
                       "per_activity": per_act,
                       "per_client_frac_heavy": {str(c): f for f, c, _, _ in per_cl},
                       "top3_clients_share_of_heavy": float(top3),
                       "test_value_rate": float((np.abs(Xm_te) > 3).mean()),
                       "test_frac_heavy": float((cnt_te >= THR).mean())}


def sec10_clients(Xm, y, sid, rep):
    title("3.2.10 Διαφορές μεταξύ των clients")
    K = int(sid.max()) + 1
    sizes = np.array([(sid == c).sum() for c in range(K)])
    print(f"  clients {K}, μεγέθη από {sizes.min()} έως {sizes.max()}, "
          f"συντελεστής μεταβλητότητας {sizes.std() / sizes.mean():.3f}")
    # απόσταση ολικής μεταβολής κάθε client από την κατανομή κλάσεων όλου του train
    glob = np.array([np.mean(y == k) for k in range(1, 7)])
    tvs, miss = [], 0
    for c in range(K):
        m = sid == c
        p = np.array([np.mean(y[m] == k) for k in range(1, 7)])
        tvs.append(np.abs(p - glob).sum() / 2)
        miss += int((p == 0).sum())
    tvs = np.array(tvs)
    print(f"  απόσταση ολικής μεταβολής: διάμεσος {np.median(tvs):.4f}, μέγιστη {tvs.max():.4f}")
    print(f"  ζεύγη (client, κλάση) χωρίς κανένα sample: {miss}")
    rep["clients"] = {"n_clients": K, "sizes": sizes.tolist(),
                      "size_cv": float(sizes.std() / sizes.mean()),
                      "tv_median": float(np.median(tvs)), "tv_max": float(tvs.max()),
                      "missing_client_class_pairs": miss}


def main():
    tr_m, te_m, meta = load_uci_har(normalize=True)
    Xm = tr_m.X.numpy().reshape(len(tr_m), -1)
    y = tr_m.y.numpy() + 1
    sid = tr_m.subject_ids
    Xm_te = te_m.X.numpy().reshape(len(te_m), -1)
    y_te = te_m.y.numpy() + 1
    Xr = np.loadtxt(os.path.join(UCI_DIR, "train", "X_train.txt"), dtype=np.float32)
    S = load_uci_signals("train")

    rep = {"dataset": "UCI HAR (Anguita et al. 2013)",
           "note": "RAW: τα αρχεία όπως διανέμονται. MODEL: μετά τη z-score με στατιστικά του train.",
           "meta": meta}
    sec1_integrity(Xr, Xm, y, sid, S, rep)
    sec2_classes(y, y_te, rep)
    sec3_per_channel(S, y, rep)
    sec4_normality(Xm, rep)
    sec5_temporal(S, y, rep)
    sec6_frequency(S, y, rep)
    sec7_channel_corr(S, rep)
    sec8_separability(Xm, y, rep)
    sec9_outliers(Xm, y, sid, Xm_te, rep)
    sec10_clients(Xm, y, sid, rep)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=_j)
    print(f"\nγράφτηκε το {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
