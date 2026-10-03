# Οι μετρικές ταξινόμησης, κοινές για όλες τις τοπολογίες.
#
# Εκτός από accuracy δίνει balanced accuracy, precision, recall και F1 σε τρεις
# σταθμίσεις (macro, weighted, micro), Cohen κ, MCC, μετρικές ανά δραστηριότητα και
# πίνακα σύγχυσης. Με macro κάθε δραστηριότητα μετράει το ίδιο, με weighted κάθε
# sample. Το micro ισούται με το accuracy σε πρόβλημα μίας ετικέτας.

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support)

ACTIVITIES = {0: "WALKING", 1: "WALKING_UPSTAIRS", 2: "WALKING_DOWNSTAIRS",
              3: "SITTING", 4: "STANDING", 5: "LAYING"}
N_CLASSES = 6


def compute_metrics(y_true, y_pred, loss=None, n_classes=N_CLASSES):
    """Όλες οι μετρικές από τις ετικέτες και τις προβλέψεις.

    Το labels=range(n_classes) κρατά τον πίνακα σύγχυσης 6x6 ακόμη κι αν ένα μοντέλο
    δεν προβλέψει ποτέ κάποια δραστηριότητα, ώστε οι πίνακες να αθροίζονται.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(n_classes))
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
           "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
           "mcc": float(matthews_corrcoef(y_true, y_pred)),
           "n_samples": int(len(y_true))}
    if loss is not None:
        out["loss"] = float(loss)
    for avg in ("macro", "weighted", "micro"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, labels=labels, zero_division=0)
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)
    out["macro_f1"] = out["f1_macro"]       # το ίδιο μέγεθος, με το όνομα που διαβάζουν τα σχήματα

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0)
    out["per_class"] = {ACTIVITIES.get(i, str(i)): {"precision": float(p[i]), "recall": float(r[i]),
                                                    "f1": float(f[i]), "support": int(sup[i])}
                        for i in labels}
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    out["class_order"] = [ACTIVITIES.get(i, str(i)) for i in labels]
    return out


def _ci95(v):
    """Διάστημα εμπιστοσύνης 95% του μέσου, με την κατανομή t."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 2:
        return [float(v[0]), float(v[0])] if n else [0.0, 0.0]
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return [float(v.mean() - h), float(v.mean() + h)]


def summarize(values, weights=None):
    """Μέσος, τυπική απόκλιση, εύρος, διάμεσος και CI95 ενός μεγέθους.

    Με βάρη (π.χ. το πλήθος samples κάθε client) δίνει και τον σταθμισμένο μέσο. Ο
    απλός μέσος απαντά τι περιμένει ένας client, ο σταθμισμένος τι περιμένει ένα sample.
    """
    v = np.asarray(values, dtype=float)
    out = {"mean": float(v.mean()),
           "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
           "min": float(v.min()), "max": float(v.max()),
           "median": float(np.median(v)),
           "ci95": _ci95(v), "n": int(len(v))}
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        out["weighted_mean"] = float(np.average(v, weights=w))
        out["weighted_std"] = float(np.sqrt(np.average((v - out["weighted_mean"]) ** 2, weights=w)))
    return out


def aggregate_runs(runs, weights=None):
    """Συγκεντρωτικά ανά μετρική από πολλά αποτελέσματα της compute_metrics.

    Οι πίνακες σύγχυσης αθροίζονται. Δίνεται και η εκδοχή τους κανονικοποιημένη
    ανά γραμμή (από τα samples κάθε δραστηριότητας, τι ποσοστό προβλέφθηκε ως τι).
    """
    if not runs:
        return {}
    scalar = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    out = {k: summarize([r[k] for r in runs], weights) for k in scalar}
    cls = list(runs[0]["per_class"])
    out["per_class"] = {c: {m: summarize([r["per_class"][c][m] for r in runs], weights)
                            for m in ("precision", "recall", "f1")}
                        for c in cls}
    for c in cls:
        out["per_class"][c]["support_total"] = int(sum(r["per_class"][c]["support"] for r in runs))
    cm = np.sum([np.asarray(r["confusion_matrix"]) for r in runs], axis=0)
    out["confusion_matrix_sum"] = cm.tolist()
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    out["confusion_matrix_rownorm"] = np.round(norm, 4).tolist()
    out["class_order"] = runs[0]["class_order"]
    out["n_runs"] = len(runs)
    return out
