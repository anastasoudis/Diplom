# Κλασικά μοντέλα ως σύγκριση για την κεντρική εκπαίδευση.
#
# Ίδια δεδομένα με το δίκτυο (load_processed), εκπαίδευση στα 7.352 samples του
# train, αξιολόγηση στα 2.947 του test, ίδιες μετρικές, ίδια seeds 0-19.
#
#   python -m implementation.centralized.classic_baselines

import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import sklearn

from implementation.centralized.classic_models import ClassicModel
from implementation.centralized.config import classic_args
from implementation.common.metrics import aggregate_runs, compute_metrics
from implementation.common.outputs import check_output
from implementation.dataset.export_processed import load_processed, provenance_block


def run_one(cm, Xtr, ytr, Xte, yte, seed):
    est = cm.build(seed)
    t0 = time.time()
    # Οι προειδοποιήσεις (π.χ. ότι ο solver σταμάτησε πριν συγκλίνει) καταγράφονται,
    # γιατί με τις προεπιλογές είναι μέρος του αποτελέσματος.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(Xtr, ytr)
    fit_s = time.time() - t0
    pred = est.predict(Xte)
    res = compute_metrics(yte, pred)
    res["train_accuracy"] = float((est.predict(Xtr) == ytr).mean())
    res["fit_seconds"] = round(fit_s, 3)
    res["seed"] = seed
    res["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})
    res["_pred_sha1"] = hashlib.sha1(np.asarray(pred, dtype=np.int64).tobytes()).hexdigest()
    return res, est


def run_model(name, args, Xtr, ytr, Xte, yte):
    cm = ClassicModel(name)
    print(f"\n  {name}: {cm!r}")
    runs, est = [], None
    for sd in args.seeds:
        r, est = run_one(cm, Xtr, ytr, Xte, yte, sd)
        runs.append(r)
        print(f"    seed {sd:>2}: test {r['accuracy']:.4f}, macro-F1 {r['f1_macro']:.4f}, "
              f"train {r['train_accuracy']:.4f} ({r['fit_seconds']:.1f} s)")
    full = aggregate_runs(runs)
    distinct = len({r["_pred_sha1"] for r in runs})
    return {"repr": repr(cm),
            "estimator_params": {k: repr(v) for k, v in est.get_params().items()},
            "seeded": cm.seeded,
            # αν τα seeds δίνουν τις ίδιες ακριβώς προβλέψεις
            "identical_across_seeds": distinct == 1,
            "distinct_prediction_sets": distinct,
            "warnings": sorted({w for r in runs for w in r["warnings"]}),
            "accuracy": full["accuracy"]["mean"], "accuracy_std": full["accuracy"]["std"],
            "macro_f1": full["f1_macro"]["mean"], "macro_f1_std": full["f1_macro"]["std"],
            "train_accuracy": full["train_accuracy"]["mean"],
            "metrics": full, "per_seed": runs, "n_seeds": len(runs)}


def main(argv=None):
    args = classic_args(argv)
    check_output(args.out, args.overwrite)
    trainset, testset, _, prov = load_processed()
    Xtr = trainset.X.numpy().reshape(len(trainset), -1)
    Xte = testset.X.numpy().reshape(len(testset), -1)
    ytr, yte = trainset.y.numpy(), testset.y.numpy()
    print(f"κλασικά μοντέλα, scikit-learn {sklearn.__version__}, train {Xtr.shape}, test {Xte.shape}")

    t0 = time.time()
    results = {m: run_model(m, args, Xtr, ytr, Xte, yte) for m in args.models}
    secs = round(time.time() - t0, 1)
    print()
    for m, r in results.items():
        print(f"  {m:<14} accuracy {r['accuracy']:.4f} ± {r['accuracy_std']:.4f}, "
              f"macro-F1 {r['macro_f1']:.4f}, διαφορετικά σύνολα προβλέψεων {r['distinct_prediction_sets']}")

    out = {"experiment": "centralized_classic_har_holdout", "args": vars(args),
           "sklearn_version": sklearn.__version__,
           "tuning": "καμία, προεπιλογές του scikit-learn",
           "results": results, "seconds": secs,
           "data_provenance": provenance_block(prov), "meta": prov["meta"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
