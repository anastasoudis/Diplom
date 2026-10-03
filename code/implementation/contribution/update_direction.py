# Δείχνει η ενημέρωση κάθε client προς την ίδια κατεύθυνση με των υπολοίπων;
#
# Για κάθε client i και γύρο μετριέται το συνημίτονο της ενημέρωσης Δ_i = M_i - M
# με τον σταθμισμένο (με τα βάρη του FedAvg) μέσο των ενημερώσεων των άλλων:
#   cos(Δ_i, Σ_{j≠i} n_j Δ_j / Σ_{j≠i} n_j)
# Το +1 σημαίνει ίδια κατεύθυνση, το 0 άσχετη και το αρνητικό αντίθετη. Είναι η
# γεωμετρική μορφή του client drift (Karimireddy κ.ά. 2020) και δεν χρησιμοποιεί το test.
# Ο μέσος στους γύρους και στα seeds συγκρίνεται με το φ. Είναι περιγραφική συσχέτιση
# σε 21 clients και δεν δείχνει αιτία.
#
# Η εκπαίδευση είναι ίδια με της ομοσπονδιακής. Το τελικό accuracy κάθε seed
# ελέγχεται απέναντι στο αποτέλεσμα του federated.py, αν υπάρχει.
#
#   python -m implementation.contribution.update_direction

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from implementation.common.helpers import resolve_device
from implementation.common.models import ModelBuilder
from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.common.train_utils import set_seed
from implementation.contribution.config import contribution_args
from implementation.dataset.export_processed import load_processed
from implementation.federated.fl.fed_utils import create_fed_clients, initialize_fed_clients
from implementation.federated.fl.server import Server

PHI_FILE = "results/contribution/har_contribution_ann_20seeds.json"
FED_FILE = "results/federated/har_fed_ann.json"


def run_seed(seed, rounds, args, trainset, testset, client_ids):
    """Μία ομοσπονδιακή εκπαίδευση. Επιστρέφει cos και ||Δ|| ανά γύρο και client."""
    set_seed(seed)
    clients = create_fed_clients(trainset, client_ids)
    model = ModelBuilder(args.model_name).build(args.dropout).to(args.device)
    clients = initialize_fed_clients(clients, args, copy.deepcopy(model))
    server = Server(args, testset, copy.deepcopy(model))
    C, N = [], []
    for _ in range(rounds):
        cap = {}

        def obs(bp, sel, _c=cap):
            _c["b"] = [np.array(a, copy=True) for a in bp]
            _c["p"] = [[np.array(a, copy=True) for a in cl.get_parameters()] for cl in sel]
            _c["n"] = [len(cl.dataset) for cl in sel]

        clients = server.update(clients, observer=obs)
        acc = server.evaluate()[0]      # όπως η ομοσπονδιακή εκπαίδευση, για την ίδια τροχιά
        D = np.array([np.concatenate([(q - b).ravel() for q, b in zip(cp, cap["b"])]) for cp in cap["p"]])
        n = np.array(cap["n"], float)
        tot, wtot = (n[:, None] * D).sum(0), n.sum()
        oth = (tot - n[:, None] * D) / (wtot - n)[:, None]       # ο μέσος των άλλων, χωρίς τον i
        nd = np.linalg.norm(D, axis=1)
        C.append(np.sum(D * oth, 1) / (nd * np.linalg.norm(oth, axis=1)))
        N.append(nd)
    return np.array(C), np.array(N), acc


def corr(x, y):
    r, p = pearsonr(x, y)
    rho, ps = spearmanr(x, y)
    return {"pearson_r": float(r), "p": float(p), "spearman_rho": float(rho), "spearman_p": float(ps)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Κατεύθυνση ενημέρωσης ανά client και φ.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--out", default="results/contribution/har_update_direction.json")
    add_overwrite_flag(ap)
    a = ap.parse_args(argv)
    check_output(a.out, a.overwrite)

    args = contribution_args(["--fl_rounds", str(a.rounds)])
    args.device = resolve_device(args.device)
    trainset, testset, client_ids, _ = load_processed()
    ref_acc = {}
    if os.path.exists(FED_FILE):
        ref_acc = {r["seed"]: r["final_accuracy"]
                   for r in json.load(open(FED_FILE, encoding="utf-8"))["results"]["subject"]["per_seed"]}
    contrib = json.load(open(PHI_FILE, encoding="utf-8"))
    scored = {r["seed"]: [pr["round"] for pr in r["shapley"]["kernelshap"]["per_round"]]
              for r in contrib["per_seed"]}
    phi = {r["client"]: r["phi_kernelshap"] for r in contrib["summary"]["clients"]}

    per_seed, gate = [], []
    for s in a.seeds:
        t0 = time.time()
        C, N, acc = run_seed(s, a.rounds, args, trainset, testset, client_ids)
        ok = abs(acc - ref_acc[s]) <= 1e-12 if s in ref_acc else None
        gate.append({"seed": s, "final_accuracy": acc, "reference": ref_acc.get(s), "match": ok})
        print(f"  seed {s:>2}: accuracy {acc:.6f}, ίδιο με την ομοσπονδιακή: {ok}, {time.time() - t0:.0f} s",
              flush=True)
        if ok is False:
            raise SystemExit(f"το seed {s} δεν ακολούθησε την τροχιά του {FED_FILE}")
        idx = np.array(scored[s]) - 1
        per_seed.append({"seed": s, "cos_all": C.mean(0).tolist(),
                         "cos_scored": C[idx].mean(0).tolist(), "norm": N.mean(0).tolist()})
    cos_all = [r["cos_all"] for r in per_seed]
    cos_sc = [r["cos_scored"] for r in per_seed]

    ids = list(range(len(cos_all[0])))
    y = np.array([phi[i] for i in ids])
    neg = y < 0
    out = {"experiment": "update_direction_vs_phi", "phi_source": PHI_FILE, "gate_reference": FED_FILE,
           "gate": gate, "seeds": [r["seed"] for r in per_seed], "rounds": a.rounds,
           "definition": "cos(Δ_i, Σ_{j≠i} n_j Δ_j / Σ_{j≠i} n_j), μέσος στους γύρους, μετά στα seeds"}
    # all_rounds: όλοι οι γύροι. scored_rounds: μόνο οι γύροι που βαθμολόγησε το φ.
    for name, M in (("all_rounds", np.mean(cos_all, 0)), ("scored_rounds", np.mean(cos_sc, 0))):
        order = np.argsort(M)
        out[name] = {"per_client": [float(v) for v in M], "corr_with_phi": corr(M, y),
                     "mean_negative": float(M[neg].mean()), "mean_positive": float(M[~neg].mean()),
                     "n_negative": int(neg.sum()), "n_positive": int((~neg).sum()),
                     "mannwhitney_p": float(mannwhitneyu(M[neg], M[~neg]).pvalue),
                     "rank_client2_from_lowest": int(np.where(order == 2)[0][0]) + 1,
                     "negatives_among_lowest9": int(neg[order[:9]].sum()),
                     "min": float(M.min()), "max": float(M.max()),
                     "per_seed_corr_with_phi": [float(pearsonr(c, y)[0])
                                                for c in (cos_all if name == "all_rounds" else cos_sc)]}
        r = out[name]["corr_with_phi"]
        print(f"  {name}: r {r['pearson_r']:+.4f} (p {r['p']:.4f}), ρ {r['spearman_rho']:+.4f}, "
              f"αρνητικοί {M[neg].mean():+.4f}, θετικοί {M[~neg].mean():+.4f}")
    nm = np.mean([r["norm"] for r in per_seed], 0)
    out["delta_norm"] = {"per_client": [float(v) for v in nm], "corr_with_phi": corr(nm, y),
                         "corr_with_cos_all": float(pearsonr(nm, np.mean(cos_all, 0))[0])}
    out["per_seed"] = per_seed
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
