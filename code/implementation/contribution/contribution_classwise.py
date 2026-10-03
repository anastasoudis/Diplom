# Σε τι συνεισφέρει κάθε client, με τιμές Shapley διανυσματικής αξίας.
#
# Ίδιο παίγνιο με το contribution.py, αλλά η αξία ενός συνασπισμού είναι διάνυσμα:
#   per_class    το recall κάθε μίας από τις 6 δραστηριότητες
#   per_subject  το accuracy σε κάθε ένα από τα 9 άτομα του test
# Το αποτέλεσμα είναι πίνακας φ με μία γραμμή ανά client και μία στήλη ανά
# δραστηριότητα ή άτομο (Tastan κ.ά. 2024, Yang κ.ά. 2024).
#
# Ο πίνακας δένεται με τις βαθμωτές τιμές μέσω μιας ταυτότητας. Αφού accuracy = Σ_c w_c v_c
# με w_c το ποσοστό της στήλης c στο test, πρέπει φ_i(accuracy) = Σ_c w_c φ_i(v_c).
# Ελέγχεται απέναντι στο αποτέλεσμα του contribution.py για τον ίδιο εκτιμητή.
#
#   python -m implementation.contribution.contribution_classwise
#   python -m implementation.contribution.contribution_classwise --value_metric per_subject \
#          --nsamples auto --gtg_max_iters 200

import copy
import json
import os
import sys
import time

import numpy as np

from implementation.common.helpers import get_criterion, resolve_device
from implementation.common.models import ModelBuilder, count_parameters
from implementation.common.outputs import check_output
from implementation.common.train_utils import set_seed
from implementation.contribution.config import contribution_args
from implementation.contribution.contribution import (capture_observer, check_matched,
                                                      make_estimators, merge_parts)
from implementation.contribution.sv.coalition import ClasswiseRoundGame
from implementation.contribution.sv.evaluate import CoalitionEvaluator
from implementation.dataset.export_processed import load_processed, provenance_block
from implementation.federated.fl.fed_utils import create_fed_clients, initialize_fed_clients
from implementation.federated.fl.server import Server

ACTS = ["WALKING", "W_UPSTAIRS", "W_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
SCALAR_RESULT = "results/contribution/har_contribution_ann_20seeds.json"


def run_one_seed(args, trainset, testset, client_ids, seed):
    set_seed(seed)
    clients = create_fed_clients(trainset, client_ids)
    sizes = [len(c.dataset) for c in clients]
    model = ModelBuilder(args.model_name).build(args.dropout).to(args.device)
    clients = initialize_fed_clients(clients, args, copy.deepcopy(model))
    server = Server(args, testset, copy.deepcopy(model))

    criterion = get_criterion(args.criterion)
    groups = testset.subject_ids if args.value_metric == "per_subject" else None
    evaluator = CoalitionEvaluator(model, testset, criterion, args.device, args.value_metric, groups)
    M, w = evaluator.n_outputs, evaluator.class_weights
    # Σε κάθε γύρο, Σ w v(N) πρέπει να είναι ακριβώς το accuracy του ίδιου μοντέλου.
    scalar_eval = CoalitionEvaluator(model, testset, criterion, args.device, "accuracy")
    rng = np.random.default_rng(args.sv_seed + seed)
    estimators = make_estimators(args, len(clients), seed)

    n_cl = len(clients)
    phi_sum = {n: np.zeros((n_cl, M)) for n in estimators}
    evals = {n: 0 for n in estimators}
    sv_seconds = {n: 0.0 for n in estimators}
    eff = {n: np.zeros(M) for n in estimators}
    rounds_scored, rounds_skipped, max_identity_err = 0, 0, 0.0
    hist, t0 = [], time.time()

    for rnd in range(1, args.fl_rounds + 1):
        cap = {}
        clients = server.update(clients, observer=capture_observer(cap))
        acc, f1, loss = server.evaluate()
        hist.append({"round": rnd, "accuracy": acc, "macro_f1": f1, "loss": loss})

        probe = ClasswiseRoundGame(cap["base"], cap["params"], cap["sizes"], evaluator, M, w)
        if max(float(np.max(np.abs(p - b))) for cp in cap["params"] for p, b in zip(cp, cap["base"])) == 0.0:
            raise AssertionError(f"γύρος {rnd}: όλες οι ενημερώσεις είναι μηδέν")
        span = probe.span()
        if abs(span) <= args.eps_between:
            rounds_skipped += 1
            continue
        rounds_scored += 1
        full = tuple([1] * len(cap["sizes"]))
        err = abs(float(w @ probe.grand_value()) - scalar_eval(probe.params_for(full)))
        max_identity_err = max(max_identity_err, err)
        if err > 1e-12:
            raise AssertionError(f"γύρος {rnd}: Σ w v(N) διαφέρει από το accuracy κατά {err:.3e}")
        for name, est in estimators.items():
            g = ClasswiseRoundGame(cap["base"], cap["params"], cap["sizes"], evaluator, M, w)
            ts = time.time()
            res = est.estimate(g, rng=np.random.default_rng(rng.integers(0, 2 ** 31 - 1)))
            sv_seconds[name] += time.time() - ts
            phi_sum[name] += res["phi"]
            evals[name] += g.n_evals
            eff[name] = np.maximum(eff[name], np.abs(res["efficiency_gap"]))

        if rnd == 1 or rnd % 10 == 0 or rnd == args.fl_rounds:
            print(f"  γύρος {rnd:>3}/{args.fl_rounds}, acc {acc:.4f}, "
                  f"βαθμολογημένοι γύροι {rounds_scored}, παραλείφθηκαν {rounds_skipped}")

    out = {"seed": seed, "n_clients": n_cl, "client_sizes": sizes,
           "final_accuracy": hist[-1]["accuracy"], "final_macro_f1": hist[-1]["macro_f1"],
           "rounds_scored": rounds_scored, "rounds_skipped": rounds_skipped,
           "class_weights": [float(x) for x in w], "output_labels": evaluator.output_labels,
           "max_round_identity_error": max_identity_err,
           "seconds": round(time.time() - t0, 1), "shapley": {}}
    for name in estimators:
        out["shapley"][name] = {"phi": phi_sum[name].tolist(), "n_evals": evals[name],
                                "seconds": round(sv_seconds[name], 1),
                                "max_efficiency_gap_per_class": [float(x) for x in eff[name]]}
    return out


def check_identity(phi_cw, w, path=SCALAR_RESULT, estimator="kernelshap"):
    """Σύγκριση του Σ_c w_c φ_i(v_c) με το φ_i(accuracy) του contribution.py."""
    if not os.path.exists(path):
        return {"checked": False, "reason": f"λείπει το {path}"}
    rows = json.load(open(path, encoding="utf-8"))["summary"]["clients"]
    key = f"phi_{estimator}"
    if key not in rows[0]:
        return {"checked": False, "reason": f"το {path} δεν έχει {key}"}
    scalar = np.array([r[key] for r in rows])
    recon = phi_cw @ np.asarray(w)
    d = np.abs(scalar - recon)
    return {"checked": True, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "scalar_sum": float(scalar.sum()), "recon_sum": float(recon.sum()),
            "reference": path, "estimator": estimator}


def print_matrix(phi, sizes, w, name, labels):
    print(f"\nφ ανά στήλη, {name}")
    print(f"  {'client':>6}{'n':>6}" + "".join(f"{a:>13}" for a in labels) + f"{'Σ w φ':>11}")
    tot = phi @ w
    for i in np.argsort(-tot):
        print(f"  {i:>6}{sizes[i]:>6}" + "".join(f"{phi[i, c]:>13.4f}" for c in range(phi.shape[1]))
              + f"{tot[i]:>11.4f}")


def main(argv=None):
    args = contribution_args(argv, "Συνεισφορά κάθε client ανά δραστηριότητα ή ανά άτομο του test.")
    args.value_metric = args.value_metric or "per_class"
    if args.value_metric not in ("per_class", "per_subject"):
        raise SystemExit("για accuracy και macro_f1 υπάρχει το contribution.py")
    args.estimators = args.estimators or ["kernelshap_cw", "gtg_cw"]
    if not all(e in ("kernelshap_cw", "gtg_cw") for e in args.estimators):
        raise SystemExit("εδώ ισχύουν μόνο οι εκτιμητές kernelshap_cw και gtg_cw")
    if args.out is None:
        args.out = ("results/contribution/har_contribution_per_subject.json" if args.value_metric == "per_subject"
                    else "results/contribution/har_contribution_classwise.json")
    check_output(args.out, args.overwrite)
    args.device = resolve_device(args.device)
    trainset, testset, client_ids, prov = load_processed()
    builder = ModelBuilder(args.model_name)
    print(f"συνεισφορά ανά στήλη ({args.value_metric}), {builder!r} με "
          f"{count_parameters(builder.build(args.dropout)):,} παραμέτρους, εκτιμητές {args.estimators}")

    if args.merge:
        runs, data_prov = merge_parts(args.merge, args)
        args.seeds = [r["seed"] for r in runs]
    else:
        runs, data_prov = [], provenance_block(prov)
        for sd in args.seeds:
            print(f"\nseed {sd}")
            runs.append(run_one_seed(args, trainset, testset, client_ids, sd))
            print(f"  τελικό accuracy {runs[-1]['final_accuracy']:.4f}, βαθμολογημένοι γύροι "
                  f"{runs[-1]['rounds_scored']}/{args.fl_rounds}, {runs[-1]['seconds']:.0f} s")

    w = np.array(runs[0]["class_weights"])
    sizes = runs[0]["client_sizes"]
    labels = ([f"S{s}" for s in runs[0]["output_labels"]] if args.value_metric == "per_subject" else ACTS)
    matched = check_matched(runs)
    summary, ident = {}, {}
    for name in args.estimators:
        phi = np.mean([np.array(r["shapley"][name]["phi"]) for r in runs], axis=0)
        summary[name] = phi
        print_matrix(phi, sizes, w, name, labels)
        ident[name] = check_identity(phi, w, estimator="kernelshap" if name.startswith("kernelshap") else "gtg")
        c = ident[name]
        print("  Σ_c w_c φ(v_c) και φ(accuracy): " + (f"μέγιστη διαφορά {c['max_abs_diff']:.2e}"
                                                        if c["checked"] else c["reason"]))

    out = {"experiment": f"contribution_{args.value_metric}_{args.model_name}_har", "args": vars(args),
           "activities": ACTS, "columns": labels, "class_weights": [float(x) for x in w],
           "client_sizes": sizes, "matched_with_headline": matched, "identity_check": ident,
           "phi": {k: v.tolist() for k, v in summary.items()}, "per_seed": runs,
           "data_provenance": data_prov}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
