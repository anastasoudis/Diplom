# Η συνεισφορά κάθε client στην ομοσπονδιακή εκπαίδευση, με τιμές Shapley.
#
# Τρέχει την ίδια ομοσπονδιακή εκπαίδευση με το federated.py (ίδια seeds, γύροι,
# μοντέλο και δεδομένα) και σε κάθε γύρο στήνει το παίγνιο του γύρου από τα τοπικά
# μοντέλα των clients (sv/coalition.py). Η τιμή κάθε client είναι το άθροισμα των
# τιμών του στους γύρους. Οι εκτιμητές βλέπουν το ίδιο v(S) και τους ίδιους γύρους,
# οπότε κάθε διαφορά τους οφείλεται στην εκτίμηση και όχι στον ορισμό.
#
#   python -m implementation.contribution.contribution                          # KernelSHAP και GTG-Tib
#   python -m implementation.contribution.contribution --value_metric macro_f1
#   python -m implementation.contribution.contribution --estimators gtg_guided
#
# Τα seeds είναι ανεξάρτητα. Για παράλληλο τρέξιμο, κάθε διεργασία παίρνει μερικά seeds
# και δικό της --out. Στο τέλος το --merge ενώνει τα αρχεία χωρίς νέο υπολογισμό.

import copy
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

from implementation.common.helpers import get_criterion, resolve_device
from implementation.common.models import ModelBuilder, count_parameters
from implementation.common.outputs import check_output
from implementation.common.train_utils import set_seed
from implementation.contribution.config import contribution_args, nsamples
from implementation.contribution.sv.coalition import RoundGame
from implementation.contribution.sv.estimator import ShapleyEstimator
from implementation.contribution.sv.estimators import kshap_l1_reg
from implementation.contribution.sv.evaluate import CoalitionEvaluator
from implementation.dataset.export_processed import (common_provenance, load_processed,
                                                     provenance_block)
from implementation.federated.fl.fed_utils import create_fed_clients, initialize_fed_clients
from implementation.federated.fl.server import Server

FED_RESULT = "results/federated/har_fed_ann.json"
MERGE_KEYS = ("value_metric", "estimators", "fl_rounds", "nsamples", "eps_between", "kshap_l1",
              "gtg_max_iters", "gtg_conv_tol", "gtg_m", "sv_seed", "epochs", "lr", "batch_size")


def make_estimators(args, n_clients, seed):
    est = {}
    for name in args.estimators:
        if name in ("kernelshap", "kernelshap_cw"):
            params = {"nsamples": nsamples(args.nsamples),
                      "l1_reg": kshap_l1_reg(args.kshap_l1, n_clients), "seed": seed}
        else:
            # Η πύλη του γύρου εφαρμόζεται έξω από τους εκτιμητές (eps_between=0 εδώ),
            # ίδια για όλους, ώστε να βλέπουν τους ίδιους γύρους.
            params = {"eps_between": 0.0, "eps_within": 1e-4,
                      "max_iters": args.gtg_max_iters, "min_iters": 10,
                      "conv_tol": args.gtg_conv_tol, "conv_window": 10}
            if name == "gtg_guided":
                params["m"] = args.gtg_m
        est[name] = ShapleyEstimator(name, params)
    return est


def capture_observer(store):
    """Κρατά τις παραμέτρους του γύρου πριν από τη συνάθροιση.

    Το get_parameters() δίνει όψεις στη μνήμη του μοντέλου, πάνω στις οποίες γράφει
    μετά η συνάθροιση. Χωρίς αντίγραφα όλα τα Δ_i θα γίνονταν μηδέν.
    """
    def observer(base_params, selected):
        store["base"] = [np.array(a, copy=True) for a in base_params]
        store["params"] = [[np.array(a, copy=True) for a in cl.get_parameters()] for cl in selected]
        store["sizes"] = [len(cl.dataset) for cl in selected]
    return observer


def run_one_seed(args, trainset, testset, client_ids, seed):
    """Μία ομοσπονδιακή εκπαίδευση, με τιμές Shapley σε κάθε γύρο για κάθε εκτιμητή."""
    set_seed(seed)
    clients = create_fed_clients(trainset, client_ids)
    sizes = [len(c.dataset) for c in clients]
    model = ModelBuilder(args.model_name).build(args.dropout).to(args.device)
    clients = initialize_fed_clients(clients, args, copy.deepcopy(model))
    server = Server(args, testset, copy.deepcopy(model))

    evaluator = CoalitionEvaluator(model, testset, get_criterion(args.criterion), args.device,
                                   metric=args.value_metric)
    rng = np.random.default_rng(args.sv_seed + seed)
    estimators = make_estimators(args, len(clients), seed)

    n_cl = len(clients)
    phi_sum = {n: np.zeros(n_cl) for n in estimators}
    per_round = {n: [] for n in estimators}
    evals = {n: 0 for n in estimators}
    sv_seconds = {n: 0.0 for n in estimators}
    rounds_scored, rounds_skipped, checked = 0, 0, False
    hist, t0 = [], time.time()

    for rnd in range(1, args.fl_rounds + 1):
        cap = {}
        clients = server.update(clients, observer=capture_observer(cap))
        acc, f1, loss = server.evaluate()
        hist.append({"round": rnd, "accuracy": acc, "macro_f1": f1, "loss": loss})

        game = RoundGame(cap["base"], cap["params"], cap["sizes"], evaluator)
        if max(float(np.max(np.abs(d[0]))) for d in game.deltas) == 0.0:
            raise AssertionError(f"γύρος {rnd}: όλες οι ενημερώσεις είναι μηδέν")
        if not checked:
            # Μία φορά ανά seed ελέγχεται ότι η ανασύνθεση του πλήρους συνασπισμού δίνει
            # το μοντέλο του server και ότι η γρήγορη αξιολόγηση δίνει ό,τι η test(). Η test()
            # περνά από DataLoader και καταναλώνει τη γεννήτρια του torch, οπότε η
            # κατάστασή της κρατιέται και επανέρχεται.
            state = torch.random.get_rng_state()
            err = game.assert_matches_fedavg(server.get_server_parameters())
            err2 = evaluator.check_matches_train_utils(server.get_server_parameters(), server.testloader)
            torch.random.set_rng_state(state)
            print(f"  ανασύνθεση και FedAvg διαφέρουν κατά {err:.1e}, "
                  f"γρήγορη αξιολόγηση και test() κατά {err2:.1e}")
            checked = True

        span = game.span()
        if abs(span) <= args.eps_between:
            rounds_skipped += 1
        else:
            rounds_scored += 1
            for name, est in estimators.items():
                g = RoundGame(cap["base"], cap["params"], cap["sizes"], evaluator)
                ts = time.time()
                res = est.estimate(g, rng=np.random.default_rng(rng.integers(0, 2 ** 31 - 1)))
                sv_seconds[name] += time.time() - ts
                phi_sum[name] += res["phi"]
                evals[name] += g.n_evals
                per_round[name].append(
                    {"round": rnd, "span": span,
                     "v_empty": float(res["v_empty"]), "v_grand": float(res["v_grand"]),
                     "phi": [float(x) for x in res["phi"]],
                     "efficiency_gap": res["efficiency_gap"],
                     "n_evals": g.n_evals, "iters": res.get("iters"), "converged": res.get("converged")})

        if rnd == 1 or rnd % 10 == 0 or rnd == args.fl_rounds:
            print(f"  γύρος {rnd:>3}/{args.fl_rounds}, acc {acc:.4f}, "
                  f"βαθμολογημένοι γύροι {rounds_scored}, παραλείφθηκαν {rounds_skipped}")

    out = {"seed": seed, "n_clients": n_cl, "client_sizes": sizes,
           "final_accuracy": hist[-1]["accuracy"], "final_macro_f1": hist[-1]["macro_f1"],
           "rounds_scored": rounds_scored, "rounds_skipped": rounds_skipped,
           "history": hist, "seconds": round(time.time() - t0, 1), "shapley": {}}
    for name in estimators:
        out["shapley"][name] = {"phi": [float(x) for x in phi_sum[name]],
                                "phi_sum": float(phi_sum[name].sum()),
                                "n_evals": evals[name], "seconds": round(sv_seconds[name], 1),
                                "per_round": per_round[name]}
    return out


def merge_parts(paths, args, keys=MERGE_KEYS):
    """Ενώνει τα per_seed παράλληλων τρεξιμάτων, αν έχουν τις ίδιες ρυθμίσεις με την εντολή."""
    parts = [json.load(open(p, encoding="utf-8")) for p in paths]
    for p, part in zip(paths, parts):
        diff = [k for k in keys if part["args"].get(k) != vars(args).get(k)]
        if diff:
            raise ValueError(f"{p}: άλλες ρυθμίσεις από την εντολή στα {diff}")
    runs = sorted((r for part in parts for r in part["per_seed"]), key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"το ίδιο seed σε περισσότερα από ένα αρχεία: {seeds}")
    return runs, common_provenance([part.get("data_provenance") for part in parts])


def check_matched(runs, ref_path=FED_RESULT):
    """Το τελικό accuracy κάθε seed πρέπει να είναι ίδιο με της ομοσπονδιακής εκπαίδευσης.

    Αλλιώς οι τιμές φ δεν αφορούν το ίδιο μοντέλο. Ο έλεγχος γίνεται αν υπάρχει το
    αποτέλεσμα του federated.py.
    """
    if not os.path.exists(ref_path):
        print(f"  δεν υπάρχει το {ref_path}, η τροχιά δεν συγκρίθηκε")
        return None
    ref = json.load(open(ref_path, encoding="utf-8"))
    by_seed = {r["seed"]: r["final_accuracy"] for r in ref["results"]["subject"]["per_seed"]}
    rows = [(r["seed"], r["final_accuracy"], by_seed[r["seed"]]) for r in runs if r["seed"] in by_seed]
    bad = [x for x in rows if abs(x[1] - x[2]) > 1e-12]
    print(f"  ίδιο τελικό accuracy με το {ref_path} σε {len(rows) - len(bad)}/{len(rows)} seeds")
    for s, a, b in bad[:5]:
        print(f"    seed {s}: εδώ {a:.6f}, στην ομοσπονδιακή εκπαίδευση {b:.6f}")
    return {"checked": len(rows), "mismatched": len(bad), "reference": ref_path}


def _rank(v):
    """Κατάταξη 1..n, 1 η μεγαλύτερη τιμή."""
    order = np.argsort(-np.asarray(v))
    r = np.empty(len(v), dtype=int)
    r[order] = np.arange(1, len(v) + 1)
    return r


def summarize(runs, estimators, client_order, sizes):
    """Μέσος και τυπική απόκλιση του φ κάθε client στα seeds, μαζί με τη συμφωνία των εκτιμητών."""
    n = len(sizes)
    out = {"n_seeds": len(runs), "clients": [], "per_estimator": {}}
    phis = {e: np.array([r["shapley"][e]["phi"] for r in runs]) for e in estimators}
    for e in estimators:
        out["per_estimator"][e] = {
            "phi_mean": [float(x) for x in phis[e].mean(axis=0)],
            "phi_std": ([float(x) for x in phis[e].std(axis=0, ddof=1)] if len(runs) > 1 else [0.0] * n),
            "total_seconds": round(sum(r["shapley"][e]["seconds"] for r in runs), 1),
            "total_evals": int(sum(r["shapley"][e]["n_evals"] for r in runs))}
    ranks = {e: _rank(out["per_estimator"][e]["phi_mean"]) for e in estimators}
    for i in range(n):
        row = {"client": int(client_order[i]), "n_samples": int(sizes[i])}
        for e in estimators:
            row[f"phi_{e}"] = out["per_estimator"][e]["phi_mean"][i]
            row[f"std_{e}"] = out["per_estimator"][e]["phi_std"][i]
            row[f"rank_{e}"] = int(ranks[e][i])
        out["clients"].append(row)

    # Kendall τ    συμφωνία στην κατάταξη, από -1 έως 1
    # Spearman ρ   συσχέτιση των κατατάξεων
    # Pearson r    συσχέτιση των τιμών
    if len(estimators) >= 2:
        out["agreement"] = {}
        for a, b in combinations(estimators, 2):
            va = np.array(out["per_estimator"][a]["phi_mean"])
            vb = np.array(out["per_estimator"][b]["phi_mean"])
            out["agreement"][f"{a}_vs_{b}"] = {
                "pearson_r": float(pearsonr(va, vb)[0]),
                "spearman_rho": float(spearmanr(va, vb)[0]),
                "kendall_tau": float(kendalltau(va, vb)[0]),
                "max_abs_diff": float(np.max(np.abs(va - vb))),
                "mean_abs_diff": float(np.mean(np.abs(va - vb))),
                "top5_overlap": int(len(set(np.argsort(-va)[:5]) & set(np.argsort(-vb)[:5])))}
    return out


def print_table(summary, estimators):
    print(f"\nσυνεισφορά ανά client ({summary['n_seeds']} seeds)")
    for row in sorted(summary["clients"], key=lambda r: -r[f"phi_{estimators[0]}"]):
        print(f"  client {row['client']:>2} ({row['n_samples']} samples)  " + "  ".join(
            f"{e} {row[f'phi_{e}']:+.5f} ± {row[f'std_{e}']:.3f} (#{row[f'rank_{e}']})" for e in estimators))
    for k, v in summary.get("agreement", {}).items():
        print(f"  {k}: Kendall τ {v['kendall_tau']:.4f}, Spearman ρ {v['spearman_rho']:.4f}, "
              f"Pearson r {v['pearson_r']:.4f}, max|Δφ| {v['max_abs_diff']:.5f}")


def main(argv=None):
    args = contribution_args(argv)
    args.estimators = args.estimators or ["kernelshap", "gtg"]
    args.value_metric = args.value_metric or "accuracy"
    if args.value_metric not in ("accuracy", "macro_f1"):
        raise SystemExit("για per_class και per_subject υπάρχει το contribution_classwise.py")
    if any(e.endswith("_cw") for e in args.estimators):
        raise SystemExit("οι εκτιμητές *_cw είναι για το contribution_classwise.py")
    if args.out is None:
        name = ("har_contribution_guided_20seeds" if args.estimators == ["gtg_guided"]
                else "har_contribution_ann_20seeds" + ("_macrof1" if args.value_metric == "macro_f1" else ""))
        args.out = f"results/contribution/{name}.json"
    check_output(args.out, args.overwrite)
    args.device = resolve_device(args.device)
    trainset, testset, client_ids, prov = load_processed()
    builder = ModelBuilder(args.model_name)
    print(f"συνεισφορά clients, {builder!r} με {count_parameters(builder.build(args.dropout)):,} "
          f"παραμέτρους, {args.fl_rounds} γύροι, αξία {args.value_metric} στο test, "
          f"εκτιμητές {args.estimators}")

    if args.merge:
        runs, data_prov = merge_parts(args.merge, args)
        args.seeds = [r["seed"] for r in runs]
        print(f"ένωση {len(args.merge)} αρχείων, seeds {args.seeds}")
    else:
        runs, data_prov = [], provenance_block(prov)
        for sd in args.seeds:
            print(f"\nseed {sd}")
            runs.append(run_one_seed(args, trainset, testset, client_ids, sd))
            r = runs[-1]
            print(f"  τελικό accuracy {r['final_accuracy']:.4f}, "
                  f"βαθμολογημένοι γύροι {r['rounds_scored']}/{args.fl_rounds}, {r['seconds']:.0f} s")

    order = sorted(set(int(c) for c in client_ids))
    summary = summarize(runs, args.estimators, order, runs[0]["client_sizes"])
    print_table(summary, args.estimators)
    matched = check_matched(runs)

    out = {"experiment": f"contribution_{args.model_name}_har", "args": vars(args),
           "matched_with_headline": matched, "summary": summary, "per_seed": runs,
           "data_provenance": data_prov, "meta": prov["meta"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"γράφτηκε το {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
