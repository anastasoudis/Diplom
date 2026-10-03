# Ορίσματα της ομοσπονδιακής εκπαίδευσης.

import argparse

from implementation.common.outputs import add_overwrite_flag


def federated_args(argv=None):
    p = argparse.ArgumentParser(description="Ομοσπονδιακή εκπαίδευση (FedAvg) στο UCI HAR.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_name", type=str, default="ann", choices=["ann"])
    p.add_argument("--epochs", type=int, default=1, help="τοπικές εποχές ανά γύρο")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--criterion", type=str, default="cross_entropy", choices=["cross_entropy"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--fl_rounds", type=int, default=100)
    # "exclude": όλοι εκτός από τους --exclude, σε κάθε γύρο (πείραμα αφαίρεσης)
    p.add_argument("--selector", type=str, default="all", choices=["all", "exclude"])
    p.add_argument("--exclude", type=int, nargs="+", default=None)
    p.add_argument("--out", type=str, default="results/federated/har_fed_ann.json")
    add_overwrite_flag(p)
    return p.parse_args(argv)
