# Shapley στην κεντρική εκπαίδευση, με παίκτες τους 21 εθελοντές (παίγνιο επανεκπαίδευσης).
#
# Το ερώτημα είναι αν οι εθελοντές με αρνητική τιμή στην ομοσπονδιακή εκπαίδευση
# βλάπτουν και όταν όλα τα δεδομένα εκπαιδεύουν ένα μοντέλο μαζί. Αν ναι, η αιτία είναι
# στα δεδομένα τους. Αν όχι, τη βλάβη τη δημιουργεί η ομοσπονδιακή συνάθροιση.
# Συγκρίνονται η κατάταξη και το πρόσημο, όχι οι τιμές, γιατί στην ομοσπονδιακή
# εκπαίδευση το φ είναι άθροισμα πολλών γύρων.
#
# Εκτιμητής είναι ο KernelSHAP με 2.090 συνασπισμούς ανά seed. Εδώ η εργασία κράτησε
# την προεπιλογή της βιβλιοθήκης l1_reg="num_features(21)", σε αντίθεση με την
# ομοσπονδιακή εκπαίδευση όπου δεν γίνεται επιλογή παικτών.
#
#   python -m implementation.contribution.data_shapley --pilot     # πρώτη και τελευταία άφιξη, 20 seeds
#   python -m implementation.contribution.data_shapley             # KernelSHAP, ένα αρχείο ανά seed
#   python -m implementation.contribution.data_shapley --summarize # σύνοψη των 20 seeds
#
# Κάθε seed θέλει περίπου 2.090 εκπαιδεύσεις, που μοιράζονται σε --workers διεργασίες.
# Αν το τρέξιμο διακοπεί, συνεχίζει από τα seeds που λείπουν.

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from implementation.centralized.config import centralized_args
from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.contribution.sv.estimators import KSHAP_L1_MODES, KernelShapEstimator, kshap_l1_reg
from implementation.contribution.sv.retrain_game import RetrainGame, init_worker
from implementation.dataset.export_processed import common_provenance, load_processed, provenance_block

CENTRAL_RESULT = "results/centralized/har_central_ann.json"
FL_PHI = "results/contribution/har_contribution_ann_20seeds.json"
PARTS = "results/contribution/data_shapley_parts"


def player_ids():
    _, _, client_ids, prov = load_processed()
    return sorted(int(c) for c in np.unique(client_ids)), provenance_block(prov)


def central_accuracy():
    """Το accuracy της κεντρικής εκπαίδευσης ανά seed, για τον έλεγχο v(N)."""
    if not os.path.exists(CENTRAL_RESULT):
        return {}
    return {r["seed"]: r["accuracy"] for r in json.load(open(CENTRAL_RESULT))["results"]["per_seed"]}


def pilot(pool, ids, seeds):
    """Πρώτη και τελευταία άφιξη κάθε εθελοντή, σε κάθε seed.

    LOO:   v(N) - v(N χωρίς i), η αλλαγή όταν ο i μπαίνει τελευταίος
    μόνος: v({i}) - v(∅), η αλλαγή όταν ο i μπαίνει πρώτος
    Ο λόγος F συγκρίνει τη διακύμανση μεταξύ εθελοντών (των μέσων τους) με τη διακύμανση
    του μέσου κάθε εθελοντή στα seeds. Μεγάλο F σημαίνει ότι οι εθελοντές ξεχωρίζουν από
    τον θόρυβο των seeds.
    """
    n = len(ids)
    loo, single, rows = np.zeros((len(seeds), n)), np.zeros((len(seeds), n)), []
    head = central_accuracy()
    for a, sd in enumerate(seeds):
        g = RetrainGame(sd, ids, pool)
        eye = np.eye(n, dtype=bool)
        t = time.time()
        v = g.values(np.vstack([np.zeros(n, bool), np.ones(n, bool), ~eye, eye]))
        v0, vN = v[0], v[1]
        loo[a] = vN - v[2:2 + n]
        single[a] = v[2 + n:] - v0
        rows.append({"seed": sd, "v_empty": v0, "v_grand": vN,
                     "grand_equals_headline": bool(vN == head.get(sd)),
                     "seconds": round(time.time() - t, 1)})
        print(f"  seed {sd}: v(∅) {v0:.4f}, v(N) {vN:.6f}, ίδιο με την κεντρική: "
              f"{rows[-1]['grand_equals_headline']}, {rows[-1]['seconds']} s", flush=True)

    def ratio(M):
        k = M.shape[0]
        within = (M.var(0, ddof=1) / k).mean()
        return float(M.mean(0).var(ddof=1) / within) if within > 0 else float("inf")

    def tstat(M):
        se = M.std(0, ddof=1) / np.sqrt(M.shape[0])
        return np.where(se > 0, M.mean(0) / se, np.inf)

    out = {"seeds": seeds, "per_seed": rows,
           "loo_mean": loo.mean(0).tolist(), "loo_std": loo.std(0, ddof=1).tolist(),
           "single_mean": single.mean(0).tolist(), "single_std": single.std(0, ddof=1).tolist(),
           "F_loo": ratio(loo), "F_single": ratio(single),
           "n_loo_abs_t_ge_2": int((np.abs(tstat(loo)) >= 2).sum()),
           "loo": loo.tolist(), "single": single.tolist()}
    print(f"F: LOO {out['F_loo']:.2f}, μόνος {out['F_single']:.2f}, "
          f"εθελοντές με |t_LOO| >= 2: {out['n_loo_abs_t_ge_2']}/{n}")
    return out


def run_seed(pool, ids, sd, nsamples, l1):
    """Ο KernelSHAP σε ένα seed του παιγνίου."""
    g = RetrainGame(sd, ids, pool)
    t = time.time()
    res = KernelShapEstimator(nsamples=nsamples, l1_reg=kshap_l1_reg(l1, len(ids)), seed=sd).estimate(g)
    ok = res["v_grand"] == central_accuracy().get(sd)
    print(f"  seed {sd}: Σφ {res['phi'].sum():+.4f}, v(N) {res['v_grand']:.6f}, ίδιο με την κεντρική: {ok}, "
          f"{g.n_evals} εκπαιδεύσεις, {time.time() - t:.0f} s", flush=True)
    return {"seed": sd, "phi": res["phi"].tolist(), "v_empty": res["v_empty"], "v_grand": res["v_grand"],
            "grand_equals_headline": bool(ok), "efficiency_gap": res["efficiency_gap"],
            "n_evals": g.n_evals, "seconds": round(time.time() - t, 1), "kshap_l1": l1}


def summarize(out_path, parts_dir):
    from scipy.stats import kendalltau, pearsonr, spearmanr
    files = sorted(f for f in os.listdir(parts_dir) if f.startswith("seed_") and f.endswith(".json"))
    runs = [json.load(open(os.path.join(parts_dir, f))) for f in files]
    if len({r["kshap_l1"] for r in runs}) != 1:
        raise ValueError(f"{parts_dir}: seeds με διαφορετικό --kshap_l1")
    prov = common_provenance([r.pop("data_provenance", None) for r in runs])
    phi = np.array([r["phi"] for r in runs])
    m = phi.mean(0)
    fl = np.asarray(json.load(open(FL_PHI))["summary"]["per_estimator"]["kernelshap"]["phi_mean"])
    res = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs], "kshap_l1": runs[0]["kshap_l1"],
           "all_grand_equal_headline": all(r["grand_equals_headline"] for r in runs),
           "phi_mean": m.tolist(), "phi_std": phi.std(0, ddof=1).tolist() if len(runs) > 1 else None,
           "negatives_centralized": [int(i) for i in np.where(m < 0)[0]],
           "negatives_fl": [int(i) for i in np.where(fl < 0)[0]],
           "sign_agreement": int((np.sign(m) == np.sign(fl)).sum()),
           "kendall_tau": float(kendalltau(m, fl)[0]), "spearman": float(spearmanr(m, fl)[0]),
           "pearson": float(pearsonr(m, fl)[0]), "per_seed": runs}
    print(f"seeds {res['n_seeds']}, v(N) ίδιο με την κεντρική σε όλα: {res['all_grand_equal_headline']}")
    print(f"αρνητικοί στην κεντρική {res['negatives_centralized']}, στην ομοσπονδιακή {res['negatives_fl']}")
    print(f"ίδιο πρόσημο {res['sign_agreement']}/21, Kendall τ {res['kendall_tau']:.3f}, "
          f"Spearman {res['spearman']:.3f}, Pearson {res['pearson']:.3f}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "data_shapley_centralized_persons", **res, "data_provenance": prov},
                  f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Shapley στην κεντρική εκπαίδευση, παίκτες οι εθελοντές.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--nsamples", default="auto")
    p.add_argument("--kshap_l1", choices=list(KSHAP_L1_MODES), default="num_features")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--parts_dir", default=PARTS)
    p.add_argument("--out", default=None, help="προεπιλογή har_data_shapley.json, "
                                                "ή har_data_shapley_firstlast_20seeds.json με --pilot")
    add_overwrite_flag(p)
    a = p.parse_args(argv)
    if a.out is None:
        a.out = ("results/contribution/har_data_shapley_firstlast_20seeds.json" if a.pilot
                 else "results/contribution/har_data_shapley.json")
    if a.summarize:
        check_output(a.out, a.overwrite)
        return summarize(a.out, a.parts_dir) or 0

    targs = centralized_args([])          # οι ρυθμίσεις της κεντρικής εκπαίδευσης, αμετάβλητες
    if a.pilot:
        check_output(a.out, a.overwrite)
    ids, prov = player_ids()
    nsamples = a.nsamples if a.nsamples == "auto" else int(a.nsamples)
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(targs,)) as pool:
        if a.pilot:
            res = pilot(pool, ids, a.seeds)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump({"experiment": "data_shapley_pilot", "train_args": vars(targs), **res,
                           "data_provenance": prov}, f, indent=2, ensure_ascii=False)
            print(f"γράφτηκε το {a.out}")
            return 0
        os.makedirs(a.parts_dir, exist_ok=True)
        for sd in a.seeds:
            path = os.path.join(a.parts_dir, f"seed_{sd:02d}.json")
            if os.path.exists(path):
                if json.load(open(path, encoding="utf-8"))["kshap_l1"] != a.kshap_l1:
                    raise SystemExit(f"το {path} έτρεξε με άλλο --kshap_l1, δώσε άλλο --parts_dir")
                print(f"  seed {sd}: υπάρχει ήδη")
                continue
            r = run_seed(pool, ids, sd, nsamples, a.kshap_l1)
            r["train_args"] = vars(targs)
            r["data_provenance"] = prov
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
