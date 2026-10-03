# Ανάγνωση των ωμών αρχείων του UCI HAR (Anguita κ.ά. 2013).
#
# Τα πειράματα δεν διαβάζουν από εδώ. Διαβάζουν τα επεξεργασμένα αρχεία του
# data/har_processed μέσω της load_processed() του export_processed.py. Το αρχείο
# αυτό το χρησιμοποιούν η εξαγωγή εκείνων των αρχείων και οι έλεγχοι στα ωμά
# δεδομένα (inspect_raw_uci.py, eda_uci.py, scaling_check.py).

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UCI_DIR = os.path.join(ROOT, "data", "UCI HAR Dataset")

ACTIVITIES = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
              4: "SITTING", 5: "STANDING", 6: "LAYING"}
N_ACTIVITIES = 6
N_FEATURES = 561

# τα 9 κανάλια των ωμών σημάτων, 128 τιμές ανά sample (2,56 s στα 50 Hz)
UCI_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z"]


class HARDataset(Dataset):
    """Samples ως τανυστές (1, 561), ή (9, 128) για τα ωμά σήματα.

    Οι ετικέτες του UCI είναι 1-6 και αποθηκεύονται ως 0-5.
    """

    def __init__(self, X, y, subject_ids=None):
        assert X.ndim == 3, f"περίμενα (N, C, L), πήρα {X.shape}"
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y - 1, dtype=torch.long)
        self.subject_ids = None if subject_ids is None else np.asarray(subject_ids)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_uci_signals(split="train", data_dir=UCI_DIR):
    """Τα ωμά σήματα ως πίνακας (N, 9, 128), με την ίδια σειρά samples με τα 561."""
    sig = [np.loadtxt(os.path.join(data_dir, split, "Inertial Signals", f"{c}_{split}.txt"),
                      dtype=np.float32)
           for c in UCI_CHANNELS]
    return np.stack(sig, axis=1)


def load_uci_har(normalize=True, data_dir=UCI_DIR):
    """Τα 561 χαρακτηριστικά, με z-score από τα στατιστικά του train.

    Επιστρέφει (trainset, testset, meta). Οι 21 εθελοντές του train γίνονται
    clients 0-20 με τη σειρά των αρχικών τους αριθμών.
    """
    def read(split):
        X = np.loadtxt(os.path.join(data_dir, split, f"X_{split}.txt"), dtype=np.float32)
        y = np.loadtxt(os.path.join(data_dir, split, f"y_{split}.txt"), dtype=np.int64)
        s = np.loadtxt(os.path.join(data_dir, split, f"subject_{split}.txt"), dtype=np.int64)
        return X, y, s

    X_train, y_train, s_train = read("train")
    X_test, y_test, _ = read("test")
    raw_range = [float(min(X_train.min(), X_test.min())), float(max(X_train.max(), X_test.max()))]

    if normalize:
        # Τα στατιστικά βγαίνουν μόνο από το train, ώστε το test να μην επηρεάζει
        # την προεπεξεργασία.
        mean = X_train.mean(axis=0)
        std = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    remap = {s: i for i, s in enumerate(sorted(set(s_train.tolist())))}
    sid = np.array([remap[s] for s in s_train.tolist()])
    meta = {
        "dataset": "UCI HAR (Anguita et al. 2013)",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_clients": len(remap),
        "raw_value_range": raw_range,
        "normalize": bool(normalize), "normalize_stats_from": "train",
        "class_counts_train": {ACTIVITIES[k]: int(v) for k, v in sorted(Counter(y_train.tolist()).items())},
        "class_counts_test": {ACTIVITIES[k]: int(v) for k, v in sorted(Counter(y_test.tolist()).items())},
        "client_sizes": [int((sid == c).sum()) for c in range(len(remap))],
        "subject_ids_original": sorted(remap),
    }
    return (HARDataset(X_train[:, None, :], y_train, sid),
            HARDataset(X_test[:, None, :], y_test),
            meta)
