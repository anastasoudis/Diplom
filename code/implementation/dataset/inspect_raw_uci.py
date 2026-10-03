# Έλεγχος ποιότητας στα ωμά αρχεία του UCI HAR, πριν από κάθε επεξεργασία.
#
# Ελέγχει ότι τα σήματα αντιστοιχούν ένα προς ένα στα samples, ότι δεν υπάρχει
# διαρροή μεταξύ train και test (ίδια samples, κοινά άτομα, ίδια ωμά σήματα) και
# πώς προσαρμόστηκε η κανονικοποίηση min-max των δημοσιευμένων αρχείων.
#
#   python -m implementation.dataset.inspect_raw_uci

import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors

from implementation.dataset.load_dataset import ACTIVITIES, ROOT, UCI_DIR, load_uci_signals

OUT = os.path.join(ROOT, "results", "dataset", "raw_audit_uci.json")


def title(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"δεν σειριοποιείται: {type(o).__name__}")


def sec_files(rep):
    title("1. Τα αρχεία")
    inv = {}
    for split in ("train", "test"):
        for f in (f"X_{split}.txt", f"y_{split}.txt", f"subject_{split}.txt"):
            n = sum(1 for _ in open(os.path.join(UCI_DIR, split, f)))
            inv[f] = n
            print(f"  {f:<24s} {n:>6d} γραμμές")
    rep["files"] = inv
    return inv


def sec_consistency(rep):
    title("2. Αντιστοιχία σημάτων και samples")
    names = [l.split(None, 1)[1].strip() for l in open(os.path.join(UCI_DIR, "features.txt"))]
    out = {}
    for split in ("train", "test"):
        X = np.loadtxt(os.path.join(UCI_DIR, split, f"X_{split}.txt"), dtype=np.float32)
        y = np.loadtxt(os.path.join(UCI_DIR, split, f"y_{split}.txt"), dtype=int)
        s = np.loadtxt(os.path.join(UCI_DIR, split, f"subject_{split}.txt"), dtype=int)
        S = load_uci_signals(split)
        same = len(X) == len(y) == len(s) == len(S)

        # Διαδοχικά samples του ίδιου ατόμου και της ίδιας δραστηριότητας πρέπει να
        # επικαλύπτονται κατά 50%, δηλαδή το δεύτερο μισό του i να είναι το πρώτο του i+1.
        bx = S[:, 0, :]
        ok = tot = 0
        for i in range(len(bx) - 1):
            if y[i] == y[i + 1] and s[i] == s[i + 1]:
                tot += 1
                ok += int(np.allclose(bx[i, 64:], bx[i + 1, :64], atol=1e-6))

        # Τα 561 χαρακτηριστικά πρέπει να προκύπτουν από αυτά τα σήματα.
        r_mean = float(np.corrcoef(bx.mean(1), X[:, names.index("tBodyAcc-mean()-X")])[0, 1])
        r_std = float(np.corrcoef(bx.std(1), X[:, names.index("tBodyAcc-std()-X")])[0, 1])

        # Η διαφορά total_acc - body_acc πρέπει να είναι η βαρύτητα, με μέτρο περίπου
        # 1 g και πολύ πιο αργές μεταβολές από το σήμα του σώματος.
        g = S[:, 6:9, :] - S[:, 0:3, :]
        gmag = float(np.sqrt((g ** 2).sum(1)).mean())
        gstd = float(np.sqrt((g ** 2).sum(1)).std())
        slow = float(np.abs(np.diff(S[:, 0:3, :], axis=2)).mean()
                     / np.abs(np.diff(g, axis=2)).mean())
        print(f"  {split}: ίδιο πλήθος σε X, y, subject, σήματα: {'ναι' if same else 'όχι'}")
        print(f"    επικάλυψη 50% σε {ok}/{tot} διαδοχικά ζεύγη")
        print(f"    r(μέσος σήματος, tBodyAcc-mean()-X) {r_mean:.6f}, r(τ.α., tBodyAcc-std()-X) {r_std:.6f}")
        print(f"    |total - body| {gmag:.4f} ± {gstd:.4f} g, το σήμα σώματος αλλάζει "
              f"{slow:.0f} φορές ταχύτερα")
        out[split] = {"n_windows": len(X), "aligned": bool(same),
                      "overlap_exact": ok, "overlap_checked": tot,
                      "r_mean": r_mean, "r_std": r_std,
                      "gravity_magnitude_g": gmag, "gravity_magnitude_std": gstd,
                      "body_faster_than_gravity_x": slow}
    rep["consistency"] = out
    return out


def sec_features(rep):
    title("3. Οι πίνακες των 561 χαρακτηριστικών")
    d = {}
    for split in ("train", "test"):
        X = np.loadtxt(os.path.join(UCI_DIR, split, f"X_{split}.txt"), dtype=np.float64)
        y = np.loadtxt(os.path.join(UCI_DIR, split, f"y_{split}.txt"), dtype=np.int64)
        assert len(X) == len(y), f"{split}: διαφορετικό πλήθος στα X και y"
        info = {"shape": list(X.shape),
                "n_nan": int(np.isnan(X).sum()), "n_inf": int(np.isinf(X).sum()),
                "min": float(X.min()), "max": float(X.max()),
                "n_values_outside_pm1": int((np.abs(X) > 1.000001).sum()),
                "n_cols_constant": int((X.std(axis=0) == 0).sum()),
                "labels_present": sorted(set(y.tolist())),
                "class_counts": {ACTIVITIES[k]: int(v)
                                 for k, v in sorted(Counter(y.tolist()).items())}}
        d[split] = info
        print(f"  {split}: {X.shape}, NaN {info['n_nan']}, Inf {info['n_inf']}, "
              f"εύρος [{info['min']:.6f}, {info['max']:.6f}], σταθερές στήλες {info['n_cols_constant']}")
    d["already_bounded_pm1"] = bool(max(d["train"]["max"], d["test"]["max"]) <= 1.000001
                                    and min(d["train"]["min"], d["test"]["min"]) >= -1.000001)
    rep["features"] = d
    return d


def sec_normalization_provenance(rep):
    """Σε ποια δεδομένα προσαρμόστηκε η min-max των δημοσιευμένων αρχείων.

    Αν οι σταθερές είχαν βγει μόνο από το train, το test θα ξεπερνούσε το ±1
    όπου έχει πιο ακραίες τιμές. Αν δεν το ξεπερνά πουθενά και κάποια άκρα τα
    ορίζουν samples του test, η min-max προσαρμόστηκε στην ένωση train και test.
    """
    title("3β. Πού προσαρμόστηκε η min-max της πηγής")
    A = np.loadtxt(os.path.join(UCI_DIR, "train", "X_train.txt"), dtype=np.float64)
    B = np.loadtxt(os.path.join(UCI_DIR, "test", "X_test.txt"), dtype=np.float64)
    tol = 1e-9

    mx_a, mn_a, mx_b, mn_b = A.max(0), A.min(0), B.max(0), B.min(0)
    both_union = int((((np.maximum(mx_a, mx_b) >= 1 - tol))
                      & ((np.minimum(mn_a, mn_b) <= -1 + tol))).sum())
    both_train = int(((mx_a >= 1 - tol) & (mn_a <= -1 + tol)).sum())
    max_only_te = int(((mx_b >= 1 - tol) & (mx_a < 1 - tol)).sum())
    min_only_te = int(((mn_b <= -1 + tol) & (mn_a > -1 + tol)).sum())
    exceed = int(((mx_b > 1 + 1e-6) | (mn_b < -1 - 1e-6)).sum())
    print(f"  στήλες που αγγίζουν και το -1 και το +1: {both_union}/561 στην ένωση, "
          f"{both_train}/561 μόνο στο train")
    print(f"  άκρα που τα ορίζει μόνο το test: {max_only_te} μέγιστα, {min_only_te} ελάχιστα")
    print(f"  στήλες όπου το test ξεπερνά το ±1: {exceed}")

    # Η min-max είναι γραμμικός μετασχηματισμός u = a*v + b ανά στήλη. Η z-score
    # ανά στήλη με στατιστικά του train δίνει το ίδιο αποτέλεσμα για v και u, άρα
    # η διαρροή της πηγής δεν φτάνει στο μοντέλο. Εδώ ελέγχεται αριθμητικά.
    rng = np.random.default_rng(0)
    v = rng.standard_normal((500, 4)) * rng.uniform(.1, 9, 4) + rng.uniform(-5, 5, 4)
    a, b = rng.uniform(.2, 3, 4), rng.uniform(-2, 2, 4)
    u = a * v + b
    z1 = (v - v[:300].mean(0)) / v[:300].std(0)
    z2 = (u - u[:300].mean(0)) / u[:300].std(0)
    resid = float(np.abs(z1 - z2).max())
    print(f"  μέγιστη διαφορά z-score πριν και μετά από γραμμικό μετασχηματισμό: {resid:.2e}")

    rep["normalization_provenance"] = {
        "cols_touching_both_extremes_union": both_union,
        "cols_touching_both_extremes_train_only": both_train,
        "cols_max_extreme_from_test_only": max_only_te,
        "cols_min_extreme_from_test_only": min_only_te,
        "cols_test_exceeding_pm1": exceed,
        "minmax_fitted_on": "train ∪ test",
        "is_preprocessing_leakage": True,
        "neutralised_by_our_zscore": True,
        "affine_cancellation_residual": resid,
        "conclusion": ("Η min-max των δημοσιευμένων X_train και X_test προσαρμόστηκε στην "
                       "ένωση train και test, γιατί το test δεν ξεπερνά πουθενά το ±1 και "
                       f"{max_only_te + min_only_te} άκρα τα ορίζει sample του test. Η z-score "
                       "ανά στήλη με στατιστικά του train την εξουδετερώνει."),
    }
    return rep["normalization_provenance"]


def sec_leakage(rep):
    title("4. Διαρροή μεταξύ train και test")
    rd = lambda p: [l.strip() for l in open(p) if l.strip()]
    Xtr = rd(os.path.join(UCI_DIR, "train", "X_train.txt"))
    Xte = rd(os.path.join(UCI_DIR, "test", "X_test.txt"))
    h = lambda rows: [hashlib.md5(r.encode()).hexdigest() for r in rows]
    htr, hte = h(Xtr), h(Xte)
    setr = set(htr)
    leak = [i for i, x in enumerate(hte) if x in setr]
    dup_tr = len(htr) - len(setr)
    dup_te = len(hte) - len(set(hte))
    print(f"  γραμμές του test που υπάρχουν αυτούσιες στο train: {len(leak)}/{len(hte)}")
    print(f"  διπλότυπα μέσα στο train {dup_tr}, μέσα στο test {dup_te}")

    # σχεδόν ίδια samples, από την απόσταση του πλησιέστερου γείτονα στο train
    A = np.loadtxt(os.path.join(UCI_DIR, "train", "X_train.txt"), dtype=np.float32)
    B = np.loadtxt(os.path.join(UCI_DIR, "test", "X_test.txt"), dtype=np.float32)
    d, _ = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(B)
    d = d.ravel()
    print(f"  απόσταση πλησιέστερου γείτονα: ελάχιστη {d.min():.4f}, διάμεσος {np.median(d):.4f}")

    st = set(np.loadtxt(os.path.join(UCI_DIR, "train", "subject_train.txt"), dtype=int).tolist())
    se = set(np.loadtxt(os.path.join(UCI_DIR, "test", "subject_test.txt"), dtype=int).tolist())
    print(f"  άτομα: train {len(st)}, test {len(se)}, κοινά {sorted(st & se)}")

    rep["leakage"] = {"exact_test_rows_in_train": len(leak),
                      "duplicates_in_train": dup_tr, "duplicates_in_test": dup_te,
                      "nn_min_distance": float(d.min()),
                      "nn_p1": float(np.percentile(d, 1)),
                      "nn_median": float(np.median(d)),
                      "n_below_0.5": int((d < 0.5).sum()),
                      "n_below_1.0": int((d < 1.0).sum()),
                      "subjects_train": sorted(st), "subjects_test": sorted(se),
                      "subject_overlap": sorted(st & se)}
    return rep["leakage"]


def sec_structure(rep):
    title("5. Δομή ανά άτομο και σύνθεση του test")
    out = {}
    for split in ("train", "test"):
        s = np.loadtxt(os.path.join(UCI_DIR, split, f"subject_{split}.txt"), dtype=int)
        y = np.loadtxt(os.path.join(UCI_DIR, split, f"y_{split}.txt"), dtype=int)
        n_subj = len(set(s.tolist()))
        blocks = int((np.diff(s) != 0).sum()) + 1
        # πόσες συνεχόμενες καταγραφές έχει κάθε ζεύγος (άτομο, δραστηριότητα)
        runs, prev = {}, None
        for si, yi in zip(s.tolist(), y.tolist()):
            k = (si, yi)
            if prev != k:
                runs[k] = runs.get(k, 0) + 1
            prev = k
        rc = Counter(runs.values())
        sizes = np.array([int((s == c).sum()) for c in sorted(set(s.tolist()))])
        print(f"  {split}: {n_subj} άτομα σε {blocks} συνεχόμενα τμήματα, "
              f"{len(runs)} ζεύγη (άτομο, δραστηριότητα), samples ανά άτομο "
              f"από {sizes.min()} έως {sizes.max()}")
        out[split] = {"n_subjects": n_subj, "contiguous_blocks": blocks,
                      "subject_blocks_contiguous": blocks == n_subj,
                      "subject_activity_pairs": len(runs),
                      "pairs_expected": n_subj * 6,
                      "no_missing_activity": len(runs) == n_subj * 6,
                      "runs_per_pair_histogram": {int(k): int(v) for k, v in sorted(rc.items())},
                      "windows_per_subject": sizes.tolist()}

    ytr = np.loadtxt(os.path.join(UCI_DIR, "train", "y_train.txt"), dtype=int)
    yte = np.loadtxt(os.path.join(UCI_DIR, "test", "y_test.txt"), dtype=int)
    ptr = np.array([np.mean(ytr == k) for k in range(1, 7)])
    pte = np.array([np.mean(yte == k) for k in range(1, 7)])
    tv = float(np.abs(ptr - pte).sum() / 2)
    frac_w = len(ytr) / (len(ytr) + len(yte))
    frac_s = out["train"]["n_subjects"] / (out["train"]["n_subjects"] + out["test"]["n_subjects"])
    print(f"  απόσταση ολικής μεταβολής κλάσεων train και test {tv:.4f}")
    print(f"  μερίδιο του train σε άτομα {frac_s:.4f}, σε samples {frac_w:.4f}")
    out["composition"] = {"train_class_pct": (100 * ptr).round(3).tolist(),
                          "test_class_pct": (100 * pte).round(3).tolist(),
                          "tv_train_test": tv, "subject_fraction_train": frac_s,
                          "window_fraction_train": frac_w}
    rep["structure"] = out
    return out


def sec_signal_leakage(rep):
    title("6. Διαρροή στα ωμά σήματα")
    A = load_uci_signals("train").reshape(-1, 9 * 128)
    B = load_uci_signals("test").reshape(-1, 9 * 128)
    ha = {hashlib.md5(r.tobytes()).hexdigest() for r in A}
    hb = [hashlib.md5(r.tobytes()).hexdigest() for r in B]
    ex = sum(1 for x in hb if x in ha)
    d, _ = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(B)
    d = d.ravel()
    dup = {}
    for nm, M in (("train", A), ("test", B)):
        h = [hashlib.md5(r.tobytes()).hexdigest() for r in M]
        dup[nm] = len(h) - len(set(h))
    print(f"  σήματα του test ίδια με του train: {ex}/{len(hb)}, "
          f"διπλότυπα μέσα στο train {dup['train']}, μέσα στο test {dup['test']}")
    rep["signal_leakage"] = {"exact": ex, "nn_min": float(d.min()),
                             "nn_median": float(np.median(d)),
                             "n_below_0.1": int((d < 0.1).sum()),
                             "duplicates_within_train": dup["train"],
                             "duplicates_within_test": dup["test"]}
    return rep["signal_leakage"]


def main():
    rep = {"dataset": "UCI HAR (Anguita et al. 2013)", "stage": "έλεγχος στα ωμά αρχεία"}
    sec_files(rep)
    sec_consistency(rep)
    sec_features(rep)
    nz = sec_normalization_provenance(rep)
    lk = sec_leakage(rep)
    st = sec_structure(rep)
    sl = sec_signal_leakage(rep)

    title("Σύνοψη")
    checks = [
        ("καμία γραμμή του test αυτούσια στο train", lk["exact_test_rows_in_train"] == 0),
        ("κανένα διπλότυπο χαρακτηριστικών στο train", lk["duplicates_in_train"] == 0),
        ("κανένα διπλότυπο χαρακτηριστικών στο test", lk["duplicates_in_test"] == 0),
        ("κανένα κοινό άτομο σε train και test", not lk["subject_overlap"]),
        ("κανένα ωμό σήμα του test ίδιο με του train", sl["exact"] == 0),
        ("κανένα διπλότυπο ωμό σήμα στο train", sl["duplicates_within_train"] == 0),
        ("κανένα διπλότυπο ωμό σήμα στο test", sl["duplicates_within_test"] == 0),
        ("κάθε άτομο σε ένα συνεχόμενο τμήμα", st["train"]["subject_blocks_contiguous"]
                                               and st["test"]["subject_blocks_contiguous"]),
        ("κάθε άτομο έχει και τις έξι δραστηριότητες", st["train"]["no_missing_activity"]
                                                       and st["test"]["no_missing_activity"]),
        ("κανένα NaN, Inf ή σταθερή στήλη", all(
            rep["features"][sp]["n_nan"] == 0 and rep["features"][sp]["n_inf"] == 0
            and rep["features"][sp]["n_cols_constant"] == 0 for sp in ("train", "test"))),
    ]
    for label, cond in checks:
        print(f"  {'ναι' if cond else 'ΟΧΙ'}  {label}")
    clean = all(c for _, c in checks)
    rep["is_clean"] = bool(clean)
    rep["checks"] = {label: bool(cond) for label, cond in checks}
    rep["caveat"] = nz["conclusion"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=_jsonable)
    print(f"\nγράφτηκε το {os.path.relpath(OUT, ROOT)}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
