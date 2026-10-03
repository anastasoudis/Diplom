# Ορίσματα της αποτίμησης συνεισφοράς στην ομοσπονδιακή εκπαίδευση.
#
# Τα ορίσματα της εκπαίδευσης έχουν τις ίδιες προεπιλογές με το federated/config.py.
# Έτσι το τρέξιμο ακολουθεί την ίδια τροχιά με την ομοσπονδιακή εκπαίδευση και οι
# τιμές φ αφορούν το ίδιο μοντέλο με το accuracy που αναφέρεται για αυτήν.

import argparse

from implementation.common.outputs import add_overwrite_flag
from implementation.contribution.sv.estimators import KSHAP_L1_MODES


def contribution_args(argv=None, description="Συνεισφορά κάθε client με τιμές Shapley."):
    p = argparse.ArgumentParser(description=description)
    # ίδια με το federated/config.py
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_name", type=str, default="ann", choices=["ann"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--criterion", type=str, default="cross_entropy", choices=["cross_entropy"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--fl_rounds", type=int, default=100)

    # η αποτίμηση
    p.add_argument("--estimators", type=str, nargs="+", default=None,
                   choices=["kernelshap", "gtg", "gtg_guided", "kernelshap_cw", "gtg_cw"])
    p.add_argument("--value_metric", type=str, default=None,
                   choices=["accuracy", "macro_f1", "per_class", "per_subject"])
    # Γύροι όπου το |v(N) - v(∅)| δεν ξεπερνά αυτό το όριο παραλείπονται (Liu κ.ά.
    # 2022, σελ. 9). Το 0,001 σε accuracy είναι περίπου 3 από τα 2.947 samples του test.
    p.add_argument("--eps_between", type=float, default=1e-3)
    p.add_argument("--kshap_l1", choices=list(KSHAP_L1_MODES), default="off",
                   help="επιλογή παικτών στον KernelSHAP (off: καμία)")
    p.add_argument("--nsamples", type=str, default="2090",
                   help="συνασπισμοί ανά γύρο για τον KernelSHAP (ακέραιος ή auto)")
    p.add_argument("--gtg_max_iters", type=int, default=100, help="μέγιστες μεταθέσεις ανά γύρο")
    p.add_argument("--gtg_conv_tol", type=float, default=0.05)
    # Το άρθρο του GTG-Shapley δεν δίνει τιμή για το m, μόνο ότι είναι πολύ μικρότερο του n.
    p.add_argument("--gtg_m", type=int, default=2, help="θέσεις με guided sampling")
    # Ξεχωριστή γεννήτρια για την επιλογή συνασπισμών και μεταθέσεων, ώστε να μην
    # αγγίζει τη γεννήτρια της εκπαίδευσης.
    p.add_argument("--sv_seed", type=int, default=12345)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--merge", type=str, nargs="+", default=None,
                   help="ένωση αποτελεσμάτων από παράλληλα τρεξίματα με διαφορετικά seeds")
    add_overwrite_flag(p)
    return p.parse_args(argv)


def nsamples(raw):
    return raw if raw == "auto" else int(raw)
