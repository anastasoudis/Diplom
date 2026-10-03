# Πόσο κοντά στο ακριβές Shapley φτάνουν οι εκτιμητές, σε λίγους γύρους του seed 0.
#
# Για κάθε ζητούμενο γύρο υπολογίζεται πρώτα το ακριβές Shapley από όλους τους
# 2^21 συνασπισμούς, περίπου 5 ώρες ανά γύρο σε έναν πυρήνα. Μετά ο KernelSHAP
# τρέχει με 128 έως 8.192 συνασπισμούς, ενώ ο GTG-Tib και ο καθοδηγούμενος GTG τρέχουν
# με 50, 100 και 200 μεταθέσεις, 5 φορές ο καθένας. Για κάθε τρέξιμο καταγράφεται η
# μέγιστη απόκλιση από το ακριβές, max_i |φ_i - φ_i ακριβές|, καθώς και ο Kendall τ με
# αυτό, δηλαδή πόσο συμφωνεί η κατάταξη των clients (1 σημαίνει ίδια σειρά).
#
# Με --m1_check προστίθεται και ο καθοδηγούμενος GTG με m = 1, με τυχαία αφετηρία
# (50/100/200 μεταθέσεις) και με σταθερή αφετηρία σε πλήρη περάσματα των 21 clients
# (42/105/210), μαζί με τον GTG-Tib στους ίδιους προϋπολογισμούς.
#
#   python -m implementation.contribution.convergence --rounds 1 --m1_check --out results/contribution/exact_r1.json
#   (το ίδιο για τους γύρους 5 και 15, σε ξεχωριστές διεργασίες)
#   python -m implementation.contribution.convergence --merge results/contribution/exact_r1.json \
#          results/contribution/exact_r5.json results/contribution/exact_r15.json
#
# Εδώ ο server δεν αξιολογεί το μοντέλο σε κάθε γύρο, όπως έτρεξε η μελέτη της
# εργασίας. Η αξιολόγηση αλλάζει τη συνέχεια της γεννήτριας, οπότε οι γύροι 5 και 15
# εδώ δεν είναι ίδιοι με τους γύρους 5 και 15 του contribution.py. Ο γύρος 1 είναι ίδιος.

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
from scipy.stats import kendalltau, pearsonr

from implementation.common.helpers import get_criterion
from implementation.common.models import ModelBuilder
from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.common.train_utils import set_seed
from implementation.contribution.config import contribution_args
from implementation.contribution.sv.coalition import RoundGame
from implementation.contribution.sv.estimator import ShapleyEstimator
from implementation.contribution.sv.evaluate import CoalitionEvaluator
from implementation.dataset.export_processed import load_processed
from implementation.federated.fl.fed_utils import create_fed_clients, initialize_fed_clients
from implementation.federated.fl.server import Server

OUT = "results/contribution/har_contribution_exact_rounds.json"


def capture_rounds(args, trainset, testset, client_ids, seed, rounds):
    """Τρέχει την ομοσπονδιακή εκπαίδευση και κρατά τα τοπικά μοντέλα των ζητούμενων γύρων."""
    set_seed(seed)
    clients = create_fed_clients(trainset, client_ids)
    model = ModelBuilder(args.model_name).build(args.dropout).to(args.device)
    clients = initialize_fed_clients(clients, args, copy.deepcopy(model))
    server = Server(args, testset, copy.deepcopy(model))
    kept = {}
    for rnd in range(1, max(rounds) + 1):
        cap = {}

        def obs(bp, sel, _c=cap):
            _c["b"] = [np.array(a, copy=True) for a in bp]
            _c["p"] = [[np.array(a, copy=True) for a in c.get_parameters()] for c in sel]
            _c["s"] = [len(c.dataset) for c in sel]
        clients = server.update(clients, observer=obs if rnd in rounds else None)
        if rnd in rounds:
            kept[rnd] = cap
    return kept, model


class _Uncached:
    """Το ίδιο παίγνιο χωρίς cache, για το ακριβές. Κάθε συνασπισμός αξιολογείται μία
    φορά, οπότε η cache δεν θα γλίτωνε τίποτα και θα κρατούσε 2^21 τιμές στη μνήμη."""

    def __init__(self, g, report_every=1 << 18):
        self.g, self.n, self.n_evals = g, g.n, 0
        self._every, self._t0 = report_every, time.time()

    def value(self, mask):
        self.n_evals += 1
        if self.n_evals % self._every == 0:
            done = self.n_evals / 2 ** self.n
            el = time.time() - self._t0
            print(f"    ακριβές: {done:.0%} σε {el / 3600:.2f} ώρες", flush=True)
        return float(self.g._evaluate(self.g.params_for(np.asarray(mask, dtype=bool))))

    def empty_value(self):
        return self.value(np.zeros(self.n, dtype=bool))

    def grand_value(self):
        return self.value(np.ones(self.n, dtype=bool))


def perm_cell(mk, ref, span, est, it, repeats, **extra):
    """Ένας εκτιμητής μεταθέσεων με `it` μεταθέσεις, `repeats` φορές με rng = default_rng(rep)."""
    taus, prs, mads, secs, nev, phis = [], [], [], [], [], []
    for rep in range(repeats):
        par = {"eps_between": 0.0, "eps_within": 1e-4, "max_iters": it, "min_iters": it,
               "conv_tol": 0.0, "conv_window": 10, **extra}
        g, t0 = mk(), time.time()
        phi = np.asarray(ShapleyEstimator(est, par).estimate(g, rng=np.random.default_rng(rep))["phi"])
        secs.append(time.time() - t0)
        nev.append(g.n_evals)
        taus.append(float(kendalltau(phi, ref)[0]))
        prs.append(float(pearsonr(phi, ref)[0]))
        mads.append(float(np.max(np.abs(phi - ref))))
        phis.append([float(x) for x in phi])
    sd = lambda v: float(np.std(v, ddof=1)) if repeats > 1 else 0.0
    return {"estimator": est, "budget": it, **extra, "n_evals": int(np.mean(nev)), "repeats": repeats,
            "seconds": round(float(np.mean(secs)), 1),
            "kendall_tau": float(np.mean(taus)), "kendall_tau_sd": sd(taus),
            "pearson_r": float(np.mean(prs)),
            "max_abs_diff": float(np.mean(mads)), "max_abs_diff_sd": sd(mads),
            "rel_max_diff": float(np.mean(mads) / max(abs(span), 1e-12)), "phi_draws": phis}


def study_round(mk, ref, span, a):
    rows = []
    for ns in a.budgets:
        g, t = mk(), time.time()
        phi = ShapleyEstimator("kernelshap", {"nsamples": ns, "l1_reg": False, "seed": 0}).estimate(g)["phi"]
        rows.append({"estimator": "kernelshap", "budget": ns, "n_evals": g.n_evals,
                     "seconds": round(time.time() - t, 1),
                     "kendall_tau": float(kendalltau(phi, ref)[0]),
                     "pearson_r": float(pearsonr(phi, ref)[0]),
                     "max_abs_diff": float(np.max(np.abs(phi - ref))),
                     "rel_max_diff": float(np.max(np.abs(phi - ref)) / max(abs(span), 1e-12))})
    for it in a.gtg_budgets:
        rows.append(perm_cell(mk, ref, span, "gtg", it, a.repeats))
    for it in a.gtg_budgets:
        rows.append(perm_cell(mk, ref, span, "gtg_guided", it, a.repeats, m=a.gtg_m))
    for r in rows:
        print(f"  {r['estimator']:<11}{r['budget']:>6}: τ {r['kendall_tau']:.4f}, "
              f"max|Δφ| {r['max_abs_diff']:.5f} ({100 * r['rel_max_diff']:.2f}% του εύρους), "
              f"{r['n_evals']} αξιολογήσεις")
    return rows


def m1_check(mk, ref, span, repeats):
    rows = [perm_cell(mk, ref, span, "gtg_guided", it, repeats, m=1, decorrelate=True)
            for it in (50, 100, 200)]
    for it in (42, 105, 210):
        rows.append(perm_cell(mk, ref, span, "gtg_guided", it, repeats, m=1, decorrelate=False))
        rows.append(perm_cell(mk, ref, span, "gtg", it, repeats))
    for r in rows:
        tag = f"m=1, decorrelate={r['decorrelate']}" if "m" in r else "GTG-Tib"
        print(f"  {tag:<28}{r['budget']:>5}: max|Δφ| {r['max_abs_diff']:.5f}, {r['n_evals']} αξιολογήσεις")
    return rows


def merge(paths, out):
    parts = [json.load(open(p, encoding="utf-8")) for p in paths]
    head = {k: parts[0][k] for k in ("experiment", "seed", "reference_estimator",
                                     "reference_coalitions", "kshap_l1")}
    for p, part in zip(paths, parts):
        if any(part[k] != v for k, v in head.items()):
            raise ValueError(f"{p}: άλλες ρυθμίσεις")
    res = dict(head, rounds=sorted(int(r) for part in parts for r in part["per_round"]),
               per_round={}, guided_m1_check={"per_round": {}}, merged_from=list(paths))
    for part in parts:
        res["per_round"].update(part["per_round"])
        res["guided_m1_check"]["per_round"].update(part.get("guided_m1_check", {}).get("per_round", {}))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Εκτιμητές Shapley απέναντι στο ακριβές, σε λίγους γύρους.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 5, 15])
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024, 2090, 8192])
    ap.add_argument("--gtg_budgets", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--gtg_m", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--m1_check", action="store_true")
    ap.add_argument("--merge", nargs="+", default=None)
    ap.add_argument("--out", default=OUT)
    add_overwrite_flag(ap)
    a = ap.parse_args(argv)
    check_output(a.out, a.overwrite)
    if a.merge:
        return merge(a.merge, a.out) or 0

    args = contribution_args(["--seeds", str(a.seed)])
    trainset, testset, client_ids, _ = load_processed()
    kept, model = capture_rounds(args, trainset, testset, client_ids, a.seed, a.rounds)
    ev = CoalitionEvaluator(model, testset, get_criterion(args.criterion), "cpu", "accuracy")

    out = {"experiment": "contribution_exact_rounds", "seed": a.seed, "rounds": a.rounds,
           "reference_estimator": "exact", "reference_coalitions": 2 ** 21, "kshap_l1": "off",
           "per_round": {}, "guided_m1_check": {"per_round": {}}}
    for rnd in a.rounds:
        cap = kept[rnd]
        mk = lambda: RoundGame(cap["b"], cap["p"], cap["s"], ev)
        span = mk().span()
        print(f"\nγύρος {rnd}, v(N) - v(∅) = {span:+.5f}")
        g, t = _Uncached(mk()), time.time()
        exact = ShapleyEstimator("exact", {"max_players": g.n}).estimate(g)
        ref, ref_secs = exact["phi"], time.time() - t
        print(f"  ακριβές: {g.n_evals:,} αξιολογήσεις σε {ref_secs:.0f} s, "
              f"Σφ - (v(N) - v(∅)) = {exact['efficiency_gap']:.1e}", flush=True)
        out["per_round"][str(rnd)] = {"span": span, "reference_phi": [float(x) for x in ref],
                                      "reference_efficiency_gap": exact["efficiency_gap"],
                                      "reference_seconds": round(ref_secs, 1),
                                      "rows": study_round(mk, ref, span, a)}
        if a.m1_check:
            out["guided_m1_check"]["per_round"][str(rnd)] = {"rows": m1_check(mk, ref, span, a.repeats)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
