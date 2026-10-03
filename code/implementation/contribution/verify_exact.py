# Ο KernelSHAP του παιγνίου επανεκπαίδευσης απέναντι στο ακριβές Shapley, σε 8 παίκτες.
#
# Στην κεντρική εκπαίδευση το φ υπολογίζεται μόνο με τον KernelSHAP. Η ιδιότητα
# efficiency (Σφ = v(N) - v(∅)) δείχνει συνέπεια αλλά όχι ακρίβεια, γιατί ισχύει και
# για λάθος κατανομή. Εδώ συγκρίνεται με τον ορισμό, με όλους τους 2^8 = 256
# συνασπισμούς των 8 πρώτων εθελοντών, για διάφορα πλήθη συνασπισμών του KernelSHAP.
# Δεν δείχνει αν οι 2.090 συνασπισμοί αρκούν για 21 παίκτες. Αυτό είναι άλλο ερώτημα.
#
#   python -m implementation.contribution.verify_exact

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy.stats import kendalltau

from implementation.centralized.config import centralized_args
from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.contribution.data_shapley import player_ids
from implementation.contribution.sv.estimators import (KSHAP_L1_MODES, ExactEstimator,
                                                       KernelShapEstimator, kshap_l1_reg)
from implementation.contribution.sv.retrain_game import RetrainGame, init_worker


def main(argv=None):
    p = argparse.ArgumentParser(description="Ακριβές Shapley και KernelSHAP σε 8 εθελοντές.")
    p.add_argument("--n_players", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--kshap_l1", choices=list(KSHAP_L1_MODES), default="num_features")
    p.add_argument("--out", default="results/contribution/har_data_shapley_exact_check.json")
    add_overwrite_flag(p)
    a = p.parse_args(argv)
    check_output(a.out, a.overwrite)

    targs = centralized_args([])
    ids, prov = player_ids()
    ids = ids[:a.n_players]           # οι πρώτοι 8 κατά αύξοντα αριθμό, ορισμένοι από πριν
    n = len(ids)
    masks = np.array([[(b >> i) & 1 for i in range(n)] for b in range(1 << n)], dtype=bool)
    print(f"παίκτες {ids}, {len(masks)} συνασπισμοί, seed {a.seed}", flush=True)

    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(targs,)) as pool:
        g = RetrainGame(a.seed, ids, pool)
        # όλοι οι συνασπισμοί με μία κλήση, ώστε να εκπαιδευτούν παράλληλα
        t = time.time()
        g.values(masks)
        warm = time.time() - t
        print(f"{g.n_evals} εκπαιδεύσεις σε {warm:.0f} s", flush=True)
        ex = ExactEstimator(max_players=n).estimate(g)
        phi_ex = np.asarray(ex["phi"])
        print(f"ακριβές: Σφ {phi_ex.sum():+.6f}, v(N) {ex['v_grand']:.6f}, v(∅) {ex['v_empty']:.6f}")
        rows = []
        for b in a.budgets:
            res = KernelShapEstimator(nsamples=b, l1_reg=kshap_l1_reg(a.kshap_l1, n), seed=a.seed).estimate(g)
            phi = np.asarray(res["phi"])
            d = float(np.max(np.abs(phi - phi_ex)))
            rows.append({"nsamples": b, "max_abs_diff": d,
                         "rel_to_span": d / (ex["v_grand"] - ex["v_empty"]),
                         "kendall_tau": float(kendalltau(phi, phi_ex)[0]),
                         "same_order": bool((np.argsort(phi) == np.argsort(phi_ex)).all()),
                         "efficiency_gap": res["efficiency_gap"], "phi": phi.tolist()})
            print(f"  KernelSHAP με {b:>3} συνασπισμούς: max|Δφ| {d:.2e} "
                  f"({100 * rows[-1]['rel_to_span']:.3f}% του εύρους), Kendall τ {rows[-1]['kendall_tau']:+.4f}")

    out = {"experiment": "exact_vs_kernelshap_centralized", "players": ids, "seed": a.seed,
           "n_coalitions": int(len(masks)), "warm_seconds": round(warm, 1),
           "exact": {"phi": phi_ex.tolist(), "v_empty": ex["v_empty"], "v_grand": ex["v_grand"],
                     "efficiency_gap": ex["efficiency_gap"]},
           "kernelshap_by_budget": rows, "kshap_l1": a.kshap_l1,
           "train_args": vars(targs), "data_provenance": prov}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
