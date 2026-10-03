# Κεντρική εκπαίδευση, με ένα μοντέλο σε όλα τα δεδομένα εκπαίδευσης και χωρίς ομοσπονδία.
#
# Εκπαίδευση στα 7.352 samples των 21 εθελοντών του train και αξιολόγηση στα 2.947
# samples των 9 εθελοντών του test. Κανένα άτομο δεν βρίσκεται και στα δύο σύνολα.
#
#   python -m implementation.centralized.centralized
#   python -m implementation.centralized.centralized --model_name logreg

import json
import os
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

from implementation.centralized.config import centralized_args
from implementation.common.helpers import get_criterion, get_optim, resolve_device
from implementation.common.metrics import aggregate_runs, compute_metrics
from implementation.common.models import ModelBuilder, count_parameters
from implementation.common.outputs import check_output
from implementation.common.train_utils import predict, set_seed, test, train
from implementation.dataset.export_processed import load_processed, provenance_block


def build(args, device):
    model = ModelBuilder(args.model_name).build(args.dropout).to(device)
    return model, get_criterion(args.criterion), get_optim(model, args.optimizer, args.lr)


def run_holdout_one(args, trainset, testset, device, seed):
    """Ένα τρέξιμο με δεδομένο seed. Εκπαιδεύονται όλα τα samples του train."""
    set_seed(seed)
    tr = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    te = DataLoader(testset, batch_size=256)
    model, crit, opt = build(args, device)

    marks = set([e for e in args.checkpoints if e <= args.epochs] + [args.epochs])
    at = {}
    for ep in range(1, args.epochs + 1):
        train(model, tr, device, crit, opt, 1)
        if ep in marks:
            yt, yp, tloss = predict(model, te, crit, device)
            met = compute_metrics(yt, yp, tloss)
            met["train_accuracy"] = test(model, tr, crit, device)[0]
            met["test_loss"] = met["loss"]
            at[ep] = met
            print(f"    εποχή {ep:>3}: train {met['train_accuracy']:.4f}, "
                  f"test {met['accuracy']:.4f}, F1 {met['f1_macro']:.4f}")

    yt, yp, tloss = predict(model, te, crit, device)
    fin = compute_metrics(yt, yp, tloss)
    fin["train_accuracy"] = test(model, tr, crit, device)[0]
    fin["test_loss"] = fin["loss"]
    fin["seed"] = seed
    fin["at_checkpoints"] = at
    fin["n_train"] = len(trainset)
    return fin


def run_holdout(args, trainset, testset, device):
    runs = []
    for sd in args.seeds:
        print(f"\n  seed {sd}")
        runs.append(run_holdout_one(args, trainset, testset, device, sd))
    std = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    marks = sorted(runs[0]["at_checkpoints"])
    curve = {m: {"test_mean": float(np.mean([r["at_checkpoints"][m]["accuracy"] for r in runs])),
                 "test_std": std([r["at_checkpoints"][m]["accuracy"] for r in runs]),
                 "train_mean": float(np.mean([r["at_checkpoints"][m]["train_accuracy"] for r in runs]))}
             for m in marks}
    full = aggregate_runs(runs)
    return {"accuracy": full["accuracy"]["mean"], "accuracy_std": full["accuracy"]["std"],
            "macro_f1": full["f1_macro"]["mean"], "macro_f1_std": full["f1_macro"]["std"],
            "train_accuracy": full["train_accuracy"]["mean"],
            "metrics": full, "per_seed": runs, "curve": curve, "n_seeds": len(runs)}


def main(argv=None):
    args = centralized_args(argv)
    check_output(args.out, args.overwrite)
    device = resolve_device(args.device)
    trainset, testset, _, prov = load_processed()
    builder = ModelBuilder(args.model_name)
    n_params = count_parameters(builder.build(args.dropout))
    print(f"κεντρική εκπαίδευση, {builder!r} με {n_params:,} παραμέτρους, "
          f"train {len(trainset)}, test {len(testset)}")

    t0 = time.time()
    res = run_holdout(args, trainset, testset, device)
    res["seconds"] = round(time.time() - t0, 1)
    print(f"\naccuracy {res['accuracy']:.4f} ± {res['accuracy_std']:.4f}, "
          f"macro-F1 {res['macro_f1']:.4f} ({len(args.seeds)} seeds, {res['seconds']} s)")

    out = {"experiment": f"centralized_{args.model_name}_har_holdout", "args": vars(args),
           "model": {"name": args.model_name, "repr": repr(builder),
                     "n_params": n_params, "uses_dropout": builder.uses_dropout},
           "results": res, "data_provenance": provenance_block(prov), "meta": prov["meta"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
