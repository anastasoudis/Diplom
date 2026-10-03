# Ατομική εκπαίδευση, όπου κάθε client εκπαιδεύει μόνος του ένα μοντέλο στα δικά του
# δεδομένα, χωρίς ομοσπονδία.
#
# Δύο ερωτήματα, με δύο διαφορετικά σύνολα αξιολόγησης:
#   Α  πόσο καλά γενικεύει σε άλλους ανθρώπους. Εκπαίδευση σε όλα τα δεδομένα του
#      client, αξιολόγηση στο κοινό test. Συγκρίνεται με την κεντρική και την
#      ομοσπονδιακή εκπαίδευση, που αξιολογούνται στο ίδιο test.
#   Β  πόσο καλά εξυπηρετεί τον ίδιο. Εκπαίδευση στο 80% των δεδομένων του,
#      αξιολόγηση στο υπόλοιπο 20% και στο κοινό test, με το ίδιο μοντέλο, ώστε η
#      διαφορά να οφείλεται μόνο στο σύνολο αξιολόγησης.
# Τα Α και Β έχουν διαφορετικό σύνολο αξιολόγησης και δεν συγκρίνονται μεταξύ τους.
#
#   python -m implementation.local_only.local_only

import json
import os
import sys
import time

import numpy as np
from torch.utils.data import DataLoader, Subset

from implementation.common.helpers import get_criterion, get_optim, resolve_device
from implementation.common.metrics import aggregate_runs, compute_metrics, summarize
from implementation.common.models import ModelBuilder, count_parameters
from implementation.common.outputs import check_output
from implementation.common.train_utils import predict, set_seed, train
from implementation.dataset.export_processed import load_processed, provenance_block
from implementation.local_only.config import local_only_args


def _wilson(k, n, z=1.96):
    """Διάστημα εμπιστοσύνης Wilson για αναλογία.

    Με περίπου 70 samples και accuracy κοντά στο 0,95 το κανονικό διάστημα ξεπερνά
    το 1, ενώ το Wilson μένει μέσα στο [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _self_split(y_own, frac):
    """Χωρίζει τα δεδομένα ενός client σε train και held-out, ανά δραστηριότητα.

    Διαδοχικά samples επικαλύπτονται κατά 50%, οπότε ένας τυχαίος διαχωρισμός θα
    έβαζε μισό κοινό σήμα και στις δύο πλευρές. Εδώ, για κάθε δραστηριότητα, το
    τελευταίο `frac` των samples της (σε χρονική σειρά) πηγαίνει στο held-out.
    """
    tr, va = [], []
    for lab in np.unique(y_own):
        pos = np.where(y_own == lab)[0]
        k = max(1, int(round(len(pos) * frac)))
        va.extend(pos[-k:].tolist())
        tr.extend(pos[:-k].tolist())
    return sorted(tr), sorted(va)


def _train_eval(args, builder, subset, eval_loaders):
    """Νέο μοντέλο, εκπαίδευση στο subset, αξιολόγηση σε κάθε loader."""
    model = builder.build(args.dropout).to(args.device)
    criterion, optimizer = get_criterion(args.criterion), get_optim(model, args.optimizer, args.lr)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
    train(model, loader, args.device, criterion, optimizer, args.epochs)
    out = {}
    for name, ld in eval_loaders.items():
        yt, yp, loss = predict(model, ld, criterion, args.device)
        out[name] = compute_metrics(yt, yp, loss)
    return out


def run_one_seed(args, trainset, testset, client_ids, seed, builder, subject_ids):
    set_seed(seed)
    global_loader = DataLoader(testset, batch_size=256)
    rows = []
    for cid in range(int(np.max(client_ids)) + 1):
        idx = np.where(client_ids == cid)[0]
        if len(idx) == 0:
            continue
        own = Subset(trainset, idx.tolist())
        a = _train_eval(args, builder, own, {"global": global_loader})["global"]

        y_own = trainset.y.numpy()[idx]
        tr_idx, va_idx = _self_split(y_own, args.self_test_size)
        b = _train_eval(args, builder, Subset(own, tr_idx),
                        {"self": DataLoader(Subset(own, va_idx), batch_size=256),
                         "global": global_loader})
        lo, hi = _wilson(round(b["self"]["accuracy"] * len(va_idx)), len(va_idx))
        rows.append({
            "client": cid, "subject_id_original": subject_ids[cid],
            "n_own": len(idx), "n_self_heldout": len(va_idx),
            "A_global_acc": a["accuracy"], "A_global_f1": a["f1_macro"],
            "A_global_f1_weighted": a["f1_weighted"],
            "A_global_balanced_acc": a["balanced_accuracy"],
            "A_global_kappa": a["cohen_kappa"], "A_global_mcc": a["mcc"],
            "B_self_acc": b["self"]["accuracy"], "B_self_f1": b["self"]["f1_macro"],
            "B_self_ci95": [lo, hi],
            "B_global_acc": b["global"]["accuracy"], "B_global_f1": b["global"]["f1_macro"],
            "B_gap_self_minus_global": b["self"]["accuracy"] - b["global"]["accuracy"],
            "A_full": a, "B_self_full": b["self"], "B_global_full": b["global"],
            "A_n_train": len(idx), "B_n_train": len(tr_idx)})
        print(f"    client {cid:>2} (n={len(idx):>3}): Α {a['accuracy']:.4f}, "
              f"Β στα δικά του {b['self']['accuracy']:.4f}, στο κοινό test {b['global']['accuracy']:.4f}")
    return rows


def _agg(rows, key):
    # απλός μέσος (κάθε client μετράει το ίδιο) και σταθμισμένος με το πλήθος samples
    return summarize([r[key] for r in rows], weights=[r["n_own"] for r in rows])


def main(argv=None):
    args = local_only_args(argv)
    check_output(args.out, args.overwrite)
    args.device = resolve_device(args.device)
    trainset, testset, client_ids, prov = load_processed()
    builder = ModelBuilder(args.model_name)
    print(f"ατομική εκπαίδευση, {builder!r} με {count_parameters(builder.build(args.dropout)):,} "
          f"παραμέτρους, {args.epochs} εποχές, held-out {args.self_test_size:.0%}")

    t0 = time.time()
    per_seed = []
    for sd in args.seeds:
        print(f"\nseed {sd}")
        per_seed.append({"seed": sd, "clients": run_one_seed(args, trainset, testset, client_ids, sd,
                                                             builder, prov["meta"]["subject_ids_original"])})
    flat = [r for s in per_seed for r in s["clients"]]
    w = [r["n_own"] for r in flat]
    res = {"n_clients": len(per_seed[0]["clients"]), "n_seeds": len(args.seeds),
           **{k: _agg(flat, k) for k in ("A_global_acc", "A_global_f1", "A_global_f1_weighted",
                                          "A_global_balanced_acc", "A_global_kappa", "A_global_mcc",
                                          "B_self_acc", "B_global_acc")},
           "B_gap": _agg(flat, "B_gap_self_minus_global"),
           "n_own": _agg(flat, "n_own"),
           "A_metrics": aggregate_runs([r["A_full"] for r in flat], weights=w),
           "B_self_metrics": aggregate_runs([r["B_self_full"] for r in flat], weights=w),
           "per_seed": per_seed}
    a = res["A_global_acc"]
    print(f"\nΑ, κοινό test: {a['mean']:.4f} ± {a['std']:.4f} (σταθμισμένος {a['weighted_mean']:.4f})")
    print(f"Β, δικά του δεδομένα: {res['B_self_acc']['mean']:.4f}, κοινό test: {res['B_global_acc']['mean']:.4f}")

    out = {"experiment": f"local_only_{args.model_name}_har", "args": vars(args),
           "results": {"subject": res}, "seconds": round(time.time() - t0, 1),
           "data_provenance": provenance_block(prov), "meta": prov["meta"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
