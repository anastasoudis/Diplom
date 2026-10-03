# Ορίσματα της ατομικής εκπαίδευσης.

import argparse

from implementation.common.outputs import add_overwrite_flag


def local_only_args(argv=None):
    p = argparse.ArgumentParser(description="Ατομική εκπαίδευση: κάθε client μόνος του.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_name", type=str, default="ann", choices=["ann"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--criterion", type=str, default="cross_entropy", choices=["cross_entropy"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--self_test_size", type=float, default=0.2,
                   help="μέρος των δεδομένων κάθε client που κρατιέται για αξιολόγηση στη μέτρηση Β")
    p.add_argument("--out", type=str, default="results/local_only/har_local_ann.json")
    add_overwrite_flag(p)
    return p.parse_args(argv)
