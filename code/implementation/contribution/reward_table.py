# Από τις τιμές φ στα μερίδια ανταμοιβής, για κάθε πολιτική του sv/reward.py.
#
# Δεν εκπαιδεύει τίποτα. Για κάθε πολιτική δίνει τα μερίδια, πόσα είναι αρνητικά ή
# μηδενικά, το μερίδιο των clients με αρνητικό φ και τον Kendall τ των μεριδίων με το
# φ (1 σημαίνει ότι η πολιτική κρατά την κατάταξη). Επίσης τι κάνει το β της
# εκθετικής και ποιες πολιτικές μένουν ίδιες αν προστεθεί μια σταθερά σε όλα τα φ.
#
#   python -m implementation.contribution.reward_table

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

from implementation.common.outputs import add_overwrite_flag, check_output
from implementation.contribution.sv.reward import RewardPolicy

PHI_FILE = "results/contribution/har_contribution_ann_20seeds.json"
LAYOUT = [("raw", None), ("zero_clip", None),
          ("exponential", {"beta": 1.0, "scale": "maxabs"}),
          ("exponential", {"beta": 1.0, "scale": "none"}),
          ("affine", {"scale": "maxabs"}), ("affine", {"scale": "minmax"}),
          ("rank", None)]
BETA_SWEEP = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def label_of(name, par):
    if name == "affine":
        return f"affine:{par['scale']}"
    if name == "exponential":
        return f"exp:b{par['beta']:g}:{par['scale']}"
    return name


def main(argv=None):
    p = argparse.ArgumentParser(description="Μερίδια ανταμοιβής από τις τιμές φ.")
    p.add_argument("--phi", type=str, default=PHI_FILE)
    p.add_argument("--estimator", type=str, default="kernelshap")
    p.add_argument("--out", type=str, default="results/contribution/har_reward_shares.json")
    add_overwrite_flag(p)
    a = p.parse_args(argv)
    check_output(a.out, a.overwrite)

    d = json.load(open(a.phi, encoding="utf-8"))
    rows = d["summary"]["clients"]
    phi = np.array([r[f"phi_{a.estimator}"] for r in rows])
    ids = [int(r["client"]) for r in rows]
    sizes = [int(r["n_samples"]) for r in rows]
    neg = [ids[i] for i in np.flatnonzero(phi < 0)]
    negmask = phi < 0
    print(f"φ από {a.phi} ({d['summary']['n_seeds']} seeds, {a.estimator}), αρνητικοί {neg}")

    res, shares = {}, {}
    for name, par in LAYOUT:
        r = RewardPolicy(name, par).allocate(phi)
        lab = label_of(name, par)
        shares[lab] = np.asarray(r["share"])
        s = shares[lab]
        res[lab] = {"policy": name, "params": par or {}, "share": [float(x) for x in s],
                    "excluded_clients": [ids[i] for i in r["excluded"]], "warnings": r["warnings"],
                    "negative_group_share": float(s[negmask].sum())}
        print(f"  {lab:<18} αρνητικά {int((s < 0).sum()):>2}, μηδενικά {int((np.abs(s) <= 1e-12).sum()):>2}, "
              f"μερίδιο αρνητικών {s[negmask].sum():.4f}, Kendall τ με το φ {kendalltau(phi, s)[0]:+.4f}")

    # Με β κοντά στο 0 τα μερίδια είναι ίσα, με μεγάλο β τα παίρνει σχεδόν όλα ο πρώτος.
    # effective_n = exp(εντροπία), δηλαδή σε πόσους ισομερείς clients αντιστοιχεί η κατανομή.
    sweep = {}
    for b in BETA_SWEEP:
        sb = np.asarray(RewardPolicy("exponential", {"beta": b, "scale": "maxabs"}).allocate(phi)["share"])
        ent = float(-np.sum(sb * np.log(sb)))
        sweep[f"{b:g}"] = {"share": [float(x) for x in sb], "ratio": float(sb.max() / sb.min()),
                           "negative_group_share": float(sb[negmask].sum()), "top_share": float(sb.max()),
                           "effective_n": float(np.exp(ent)),
                           "kendall_tau_with_phi": float(kendalltau(phi, sb)[0])}
        print(f"  β {b:>5g}: λόγος άκρων {sb.max() / sb.min():6.1f}, μερίδιο αρνητικών {sb[negmask].sum():.4f}, "
              f"ισοδύναμοι ισομερείς {np.exp(ent):.1f}")

    # ποιες πολιτικές δεν αλλάζουν όταν προστεθεί 0,5 σε όλα τα φ
    shift = {}
    for name, par in LAYOUT:
        lab = label_of(name, par)
        s2 = np.asarray(RewardPolicy(name, par).allocate(phi + 0.5)["share"])
        shift[lab] = float(np.abs(s2 - shares[lab]).max())

    out = {"source": a.phi, "estimator": a.estimator, "n_seeds": d["summary"]["n_seeds"],
           "clients": ids, "n_samples": sizes, "phi": [float(x) for x in phi],
           "negative_clients": neg, "policies": res, "beta_sweep": sweep,
           "shift_invariance_max_delta": shift,
           "notes": ["Προτάσεις: exponential και affine (Tastan κ.ά. 2024, Εξ. 6).",
                     "Βάσεις σύγκρισης: raw, zero_clip, rank."]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"γράφτηκε το {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
