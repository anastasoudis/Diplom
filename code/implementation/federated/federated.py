# Ομοσπονδιακή εκπαίδευση με FedAvg στο UCI HAR.
#
# Clients είναι οι 21 εθελοντές του train, ο καθένας με τα δικά του samples. Σε κάθε
# γύρο όλοι εκπαιδεύουν τοπικά το global μοντέλο και ο server παίρνει τον μέσο όρο
# των παραμέτρων τους, σταθμισμένο με το πλήθος των samples. Αξιολόγηση στα 2.947
# samples των 9 εθελοντών του test, που δεν είναι ποτέ clients.
#
#   python -m implementation.federated.federated
#   python -m implementation.federated.federated --fl_rounds 20 --epochs 5 --seeds 0 1 2 3 4 \
#          --out results/federated/sweep_fedE_r20_e5.json

import copy
import json
import os
import sys
import time

import numpy as np
import torch

from implementation.common.helpers import resolve_device
from implementation.common.metrics import aggregate_runs
from implementation.common.models import ModelBuilder, count_parameters
from implementation.common.outputs import check_output
from implementation.common.train_utils import set_seed
from implementation.dataset.export_processed import load_processed, provenance_block
from implementation.federated.config import federated_args
from implementation.federated.fl.fed_utils import create_fed_clients, initialize_fed_clients
from implementation.federated.fl.server import Server


def run_one(args, trainset, testset, client_ids, seed):
    set_seed(seed)
    clients = create_fed_clients(trainset, client_ids)
    sizes = [len(c.dataset) for c in clients]
    model = ModelBuilder(args.model_name).build(args.dropout).to(args.device)
    clients = initialize_fed_clients(clients, args, copy.deepcopy(model))
    server = Server(args, testset, copy.deepcopy(model))

    hist, t0 = [], time.time()
    for rnd in range(1, args.fl_rounds + 1):
        clients = server.update(clients)
        acc, f1, loss = server.evaluate()
        hist.append({"round": rnd, "accuracy": acc, "macro_f1": f1, "loss": loss})
        if rnd == 1 or rnd % 10 == 0 or rnd == args.fl_rounds:
            print(f"  γύρος {rnd:>3}/{args.fl_rounds}, acc {acc:.4f}, f1 {f1:.4f}, loss {loss:.4f}")

    best = max(hist, key=lambda h: h["accuracy"])
    out = server.evaluate_full()
    # Accuracy ανά άτομο του test, με forward χωρίς DataLoader, ώστε να μη
    # χρησιμοποιηθεί η γεννήτρια. Ο σταθμισμένος μέσος τους πρέπει να δίνει το
    # συνολικό accuracy.
    server.model.eval()
    with torch.no_grad():
        pred = server.model(testset.X.to(args.device)).argmax(1).cpu().numpy()
    sid, yt = np.asarray(testset.subject_ids), testset.y.numpy()
    per_s = {int(s): float((pred[sid == s] == yt[sid == s]).mean()) for s in np.unique(sid)}
    n_s = {int(s): int((sid == s).sum()) for s in np.unique(sid)}
    pooled = sum(per_s[s] * n_s[s] for s in per_s) / sum(n_s.values())
    if abs(pooled - out["accuracy"]) > 1e-9:
        raise AssertionError(f"το accuracy ανά άτομο δεν δίνει το συνολικό: {pooled} και {out['accuracy']}")
    out.update({"per_test_subject_accuracy": per_s, "per_test_subject_n": n_s,
                "seed": seed, "n_clients": len(clients), "client_sizes": sizes,
                "final_accuracy": hist[-1]["accuracy"], "final_macro_f1": hist[-1]["macro_f1"],
                "best_accuracy": best["accuracy"], "best_round": best["round"],
                "history": hist, "seconds": round(time.time() - t0, 1)})
    print(f"  τελικό accuracy {out['accuracy']:.4f}, macro-F1 {out['f1_macro']:.4f}")
    return out


def run_all(args, trainset, testset, client_ids):
    runs = []
    for sd in args.seeds:
        print(f"\nseed {sd}")
        runs.append(run_one(args, trainset, testset, client_ids, sd))
    acc = [r["final_accuracy"] for r in runs]
    f1 = [r["final_macro_f1"] for r in runs]
    std = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"n_clients": runs[0]["n_clients"], "client_sizes": runs[0]["client_sizes"],
            "n_seeds": len(runs), "metrics": aggregate_runs(runs),
            "final_accuracy": float(np.mean(acc)), "final_accuracy_std": std(acc),
            "final_macro_f1": float(np.mean(f1)), "final_macro_f1_std": std(f1),
            "best_accuracy": float(np.mean([r["best_accuracy"] for r in runs])),
            "best_round": int(np.median([r["best_round"] for r in runs])),
            "per_seed": runs, "seconds": round(sum(r["seconds"] for r in runs), 1)}


def main(argv=None):
    args = federated_args(argv)
    check_output(args.out, args.overwrite)
    args.device = resolve_device(args.device)
    trainset, testset, client_ids, prov = load_processed()
    builder = ModelBuilder(args.model_name)
    print(f"ομοσπονδιακή εκπαίδευση, {builder!r} με {count_parameters(builder.build(args.dropout)):,} "
          f"παραμέτρους, {args.fl_rounds} γύροι, {args.epochs} τοπικές εποχές")

    res = run_all(args, trainset, testset, client_ids)
    print(f"\nτελικό accuracy {res['final_accuracy']:.4f} ± {res['final_accuracy_std']:.4f}, "
          f"macro-F1 {res['final_macro_f1']:.4f}")
    # Το "subject" λέει ότι clients είναι οι εθελοντές, όπως τους δίνει το dataset.
    out = {"experiment": f"federated_{args.model_name}_har", "args": vars(args),
           "results": {"subject": res},
           "data_provenance": provenance_block(prov), "meta": prov["meta"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
