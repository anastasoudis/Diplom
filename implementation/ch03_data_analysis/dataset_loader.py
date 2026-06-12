# =============================================================================
# ch03_data_analysis / dataset_loader.py
# -----------------------------------------------------------------------------
# Φόρτωση και προ-επεξεργασία του AAL dataset για χρήση σε FL.
#
# Pipeline:
#   αρχείο .txt → numpy arrays → Butterworth φίλτρο → Z-score → HARDataset
#
# Αντιστοιχεί σε: §3.3 (Preprocessing Pipeline) της διπλωματικής
# Χρησιμοποιείται από: ch04_fl_implementation/client.py
# =============================================================================

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Literal


# --- Παράμετροι dataset και preprocessing ------------------------------------

WINDOW_SIZE    = 128    # δείγματα ανά παράθυρο (= 2.56 s @ 50 Hz)
OVERLAP        = 0.5    # 50% επικάλυψη → stride = 64 δείγματα
SAMPLING_RATE  = 50     # Hz — συχνότητα δειγματοληψίας smartphone
N_CLASSES      = 6      # WALKING, W_UPSTAIRS, W_DOWNSTAIRS, SITTING, STANDING, LAYING

# Παράμετροι Butterworth — χωρίζει βαρύτητα (< 0.3 Hz) από κίνηση σώματος (> 0.3 Hz)
BUTTER_ORDER  = 3
BUTTER_CUTOFF = 0.3   # Hz

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


# --- Φιλτράρισμα σήματος -----------------------------------------------------

def butterworth_lowpass(signal: np.ndarray, cutoff=BUTTER_CUTOFF,
                        fs=SAMPLING_RATE, order=BUTTER_ORDER) -> np.ndarray:
    """Εφαρμόζει low-pass Butterworth φίλτρο για εξαγωγή συνιστώσας βαρύτητας.

    Οτιδήποτε κάτω από cutoff Hz θεωρείται σταθερή βαρύτητα (DC component).
    Χρησιμοποιεί filtfilt (zero-phase) ώστε να μην αλλοιωθεί η χρονική δομή.
    """
    nyq = fs / 2.0   # Nyquist: ανώτατη συχνότητα που μπορούμε να "δούμε" @ fs Hz

    # butter() επιστρέφει τους συντελεστές b (zeros) και a (poles) του φίλτρου
    b, a = butter(order, cutoff / nyq, btype="low")

    if signal.ndim == 1:
        # zero-phase: εφαρμόζει φίλτρο μια φορά forward, μια backward
        # αποτέλεσμα: καθυστέρηση φάσης = 0, effective order = 2×order
        return filtfilt(b, a, signal)

    # αν έχει πολλά κανάλια (π.χ. X,Y,Z), φιλτράρισε κάθε στήλη ξεχωριστά
    return np.column_stack([filtfilt(b, a, signal[:, c])
                            for c in range(signal.shape[1])])


def extract_body_acc(acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Χωρίζει το σήμα επιταχυνσιόμετρου σε βαρύτητα + κίνηση σώματος.

    Επιστρέφει (body_acc, gravity) — και τα δύο shape (T, 3).
    """
    gravity = butterworth_lowpass(acc)
    body    = acc - gravity   # η "καθαρή" κίνηση χωρίς την επίδραση βαρύτητας
    return body, gravity


# --- Κανονικοποίηση ----------------------------------------------------------

def zscore_normalize(X: np.ndarray,
                     mean: np.ndarray | None = None,
                     std:  np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score κανονικοποίηση: (x - μ) / σ ανά feature/κανάλι.

    Αν δοθεί mean/std, τα χρησιμοποιεί (π.χ. statistics από training set).
    Αν όχι, τα υπολογίζει από τα X — κατάλληλο μόνο για training data.

    Η κανονικοποίηση γίνεται ανά client (δεν κοινοποιούνται τα statistics
    στον server) — βασική αρχή privacy στο FL.
    """
    if mean is None:
        mean = X.mean(axis=0)

    if std is None:
        std = X.std(axis=0)
        # αν κάποιο feature είναι σταθερό (std=0), βάζουμε 1 για να αποφύγουμε div/0
        std = np.where(std == 0, 1.0, std)

    return (X - mean) / std, mean, std


# --- Sliding window ----------------------------------------------------------

def sliding_window(signal: np.ndarray, labels: np.ndarray,
                   win_size=WINDOW_SIZE, overlap=OVERLAP
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Κόβει χρονοσειρά σε παράθυρα σταθερού μεγέθους με επικάλυψη.

    Παράδειγμα: 3000 δείγματα, win=128, stride=64 → 45 παράθυρα.
    Το label κάθε παραθύρου είναι η πλειοψηφική κλάση (majority vote).

    Επιστρέφει:
        windows      shape (N, C, win_size)  — C = αριθμός καναλιών
        win_labels   shape (N,)              — label ανά παράθυρο
    """
    stride = int(win_size * (1.0 - overlap))   # π.χ. 128 * 0.5 = 64
    windows, win_labels = [], []

    for start in range(0, len(signal) - win_size + 1, stride):
        end = start + win_size
        win = signal[start:end].T           # transpose: (T, C) → (C, T)
        lab_seg = labels[start:end]
        # majority vote: βρες ποια κλάση εμφανίζεται πιο συχνά στο παράθυρο
        majority = Counter(lab_seg.tolist()).most_common(1)[0][0]
        windows.append(win)
        win_labels.append(majority)

    if not windows:
        return (np.empty((0, signal.shape[1], win_size)),
                np.empty(0, dtype=int))

    return np.array(windows, dtype=np.float32), np.array(win_labels, dtype=np.int64)


# --- Dataset class -----------------------------------------------------------

class HARDataset(Dataset):
    """PyTorch Dataset που αναδιπλώνει τα παράθυρα HAR για χρήση σε DataLoader.

    Κάθε στοιχείο: (X_window, label) με X_window ∈ R^(C × 128).
    Τα labels μετατρέπονται από 1-indexed (1–6) σε 0-indexed (0–5)
    για το CrossEntropyLoss του PyTorch.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 subject_ids: np.ndarray | None = None):
        # μετατροπή σε PyTorch tensors
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy((y - 1).astype(np.int64))  # 1-indexed → 0-indexed

        # subject_ids: ποιος χρήστης έκανε κάθε παράθυρο (για FL split)
        self.subject_ids = subject_ids if subject_ids is not None else np.zeros(len(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def input_shape(self):
        """Επιστρέφει (C, T) — channels × timesteps."""
        return tuple(self.X.shape[1:])

    def class_distribution(self) -> dict[int, int]:
        """Πόσα παράθυρα υπάρχουν ανά κλάση (0-indexed)."""
        return dict(sorted(Counter(self.y.tolist()).items()))


# --- Φόρτωση dataset ---------------------------------------------------------

def _infer_subject_ids(y: np.ndarray, n_activities: int = 6) -> np.ndarray:
    """Εξάγει subject IDs από τη block-sorted δομή των labels.

    Τα δεδομένα είναι αποθηκευμένα ως:
        subject_0: [act_1×N₁, act_2×N₂, ..., act_6×N₆]
        subject_1: [act_1×M₁, ..., act_6×M₆]
        ...
    Βρίσκουμε τα "runs" (συνεχόμενες ίδιες τιμές) και ομαδοποιούμε
    κάθε 6 runs σε έναν subject.
    """
    # groupby: βρες συνεχόμενες ομάδες ίδιων τιμών
    runs = [(k, sum(1 for _ in g)) for k, g in groupby(y.tolist())]

    subject_ids = np.zeros(len(y), dtype=int)
    idx  = 0
    subj = 0
    for block_start in range(0, len(runs), n_activities):
        block = runs[block_start:block_start + n_activities]
        for _, count in block:
            subject_ids[idx:idx + count] = subj
            idx += count
        subj += 1

    return subject_ids


def load_aal(data_dir: str | Path,
                  normalize: bool = True
                  ) -> tuple[HARDataset, HARDataset, dict]:
    """Φορτώνει το AAL dataset από τα αρχεία .txt.

    Επιστρέφει (train_dataset, test_dataset, stats) όπου stats περιέχει
    αριθμούς δειγμάτων, κατανομή κλάσεων και παραμέτρους κανονικοποίησης.

    Σημείωση: η κανονικοποίηση υπολογίζεται ΜΟΝΟ στο training set
    και εφαρμόζεται στο test — αποφεύγουμε data leakage.
    """
    data_dir = Path(data_dir)
    print(f"Loading dataset from: {data_dir}")

    # φόρτωση feature matrices (N × 561) και labels (N,)
    X_train = np.loadtxt(data_dir / "final_X_train.txt", delimiter=",", dtype=np.float32)
    y_train = np.loadtxt(data_dir / "final_y_train.txt", delimiter=",", dtype=np.int64)
    X_test  = np.loadtxt(data_dir / "final_X_test.txt",  delimiter=",", dtype=np.float32)
    y_test  = np.loadtxt(data_dir / "final_y_test.txt",  delimiter=",", dtype=np.int64)

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples")

    # υπολογισμός statistics μόνο από training set
    mean = std = None
    if normalize:
        mean = X_train.mean(axis=0)       # μέσος ανά feature
        std  = X_train.std(axis=0)
        std  = np.where(std == 0, 1.0, std)   # αποφυγή διαίρεσης με 0

        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std  # ίδια statistics — δεν "βλέπουμε" το test

    # reshape σε (N, 1, 561): 1 κανάλι × 561 feature "θέσεις"
    # έτσι το Conv1d βλέπει L=561 temporal positions αντί 561 κανάλια
    X_train_3d = X_train[:, np.newaxis, :]
    X_test_3d  = X_test[:, np.newaxis, :]

    # εξαγωγή subject IDs από τη δομή των labels
    subject_ids = _infer_subject_ids(y_train)
    n_subjects  = int(subject_ids.max()) + 1
    print(f"  Training subjects identified: {n_subjects}")

    stats = {
        "n_train":   len(y_train),
        "n_test":    len(y_test),
        "n_subjects_train": n_subjects,
        "mean":      mean,
        "std":       std,
        "class_counts_train": dict(sorted(Counter(y_train.tolist()).items())),
        "class_counts_test":  dict(sorted(Counter(y_test.tolist()).items())),
    }

    train_ds = HARDataset(X_train_3d, y_train, subject_ids=subject_ids)
    test_ds  = HARDataset(X_test_3d,  y_test)

    return train_ds, test_ds, stats


def load_uci_har_features(data_dir: str | Path,
                          normalize: bool = True
                          ) -> tuple[HARDataset, HARDataset, dict]:
    """Φορτώνει το UCI HAR στην 561-feature αναπαράσταση (όχι raw signals).

    Σκοπός: ΔΙΚΑΙΗ σύγκριση με το AAL dataset χρησιμοποιώντας το ΙΔΙΟ 1D-CNN
    μοντέλο (input 1×561). Διαβάζει τα έτοιμα X_train.txt/X_test.txt (561
    engineered features) του standard UCI HAR αντί των Inertial Signals.
    Τα labels (1–6) τα κάνει 0-indexed η HARDataset.
    """
    data_dir = Path(data_dir)
    print(f"Loading UCI HAR (561-feature) from: {data_dir}")

    X_train = np.loadtxt(data_dir / "train" / "X_train.txt",       dtype=np.float32)
    y_train = np.loadtxt(data_dir / "train" / "y_train.txt",       dtype=np.int64)
    subj    = np.loadtxt(data_dir / "train" / "subject_train.txt", dtype=int)
    X_test  = np.loadtxt(data_dir / "test"  / "X_test.txt",        dtype=np.float32)
    y_test  = np.loadtxt(data_dir / "test"  / "y_test.txt",        dtype=np.int64)

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples")

    if normalize:
        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        std  = np.where(std == 0, 1.0, std)
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std

    X_train_3d = X_train[:, np.newaxis, :]   # (N, 1, 561)
    X_test_3d  = X_test[:,  np.newaxis, :]

    uniq  = np.unique(subj)
    remap = {s: i for i, s in enumerate(uniq)}
    subj0 = np.array([remap[s] for s in subj], dtype=int)

    stats = {
        "n_train":    int(len(y_train)),
        "n_test":     int(len(y_test)),
        "n_subjects": int(len(uniq)),
        "n_classes":  6,
        "class_counts_train": dict(sorted(Counter(y_train.tolist()).items())),
    }

    train_ds = HARDataset(X_train_3d, y_train, subject_ids=subj0)
    test_ds  = HARDataset(X_test_3d,  y_test)
    return train_ds, test_ds, stats


# --- UCI HAR loader (raw inertial signals, 9 channels × 128 samples) ---------

UCI_HAR_CHANNELS = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _load_uci_split(split_dir: Path, split: Literal["train", "test"]) -> tuple:
    """Φορτώνει ένα split (train ή test) του UCI HAR από τα Inertial Signals.

    Κάθε αρχείο έχει shape (N, 128) — Ν παράθυρα × 128 samples.
    Τα 9 κανάλια στοιβάζονται σε (N, 9, 128) για 1D-CNN input.
    """
    signals_dir = split_dir / "Inertial Signals"
    channels = []
    for ch in UCI_HAR_CHANNELS:
        fp = signals_dir / f"{ch}_{split}.txt"
        channels.append(np.loadtxt(fp, dtype=np.float32))

    X = np.stack(channels, axis=1)                                       # (N, 9, 128)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64)          # (N,)
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=int)  # (N,)
    return X, y, subjects


def load_uci_har(data_dir: str | Path,
                 normalize: bool = True
                 ) -> tuple[HARDataset, HARDataset, dict]:
    """Φορτώνει το UCI HAR dataset (raw inertial, 9 κανάλια × 128 samples).

    Η κανονικοποίηση είναι per-channel z-score με statistics μόνο από train.
    Τα subject IDs είναι διαθέσιμα απευθείας (όχι inferred).
    """
    data_dir = Path(data_dir)
    print(f"Loading UCI HAR from: {data_dir}")

    X_train, y_train, subj_train = _load_uci_split(data_dir / "train", "train")
    X_test,  y_test,  subj_test  = _load_uci_split(data_dir / "test",  "test")

    print(f"  Train: {X_train.shape} ({len(np.unique(subj_train))} subjects)")
    print(f"  Test:  {X_test.shape}  ({len(np.unique(subj_test))} subjects)")

    mean = std = None
    if normalize:
        # mean/std ανά κανάλι — reduce across (N, T)
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std  = X_train.std(axis=(0, 2),  keepdims=True)
        std  = np.where(std == 0, 1.0, std)

        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std

    # επαναφορά subject_ids σε 0-based για συμβατότητα με create_fl_splits
    uniq_train = np.unique(subj_train)
    remap      = {s: i for i, s in enumerate(uniq_train)}
    subj_train_0 = np.array([remap[s] for s in subj_train], dtype=int)

    stats = {
        "n_train": len(y_train),
        "n_test":  len(y_test),
        "n_subjects_train": int(len(uniq_train)),
        "n_subjects_test":  int(len(np.unique(subj_test))),
        "subjects_train":   uniq_train.tolist(),
        "subjects_test":    np.unique(subj_test).tolist(),
        "mean": mean,
        "std":  std,
        "class_counts_train": dict(sorted(Counter(y_train.tolist()).items())),
        "class_counts_test":  dict(sorted(Counter(y_test.tolist()).items())),
        "n_channels": X_train.shape[1],
        "window_size": X_train.shape[2],
    }

    train_ds = HARDataset(X_train, y_train, subject_ids=subj_train_0)
    test_ds  = HARDataset(X_test,  y_test)
    return train_ds, test_ds, stats


# --- FL splits ---------------------------------------------------------------

def create_fl_splits(
    dataset: HARDataset,
    strategy: Literal["by_subject", "iid", "dirichlet"] = "by_subject",
    n_clients: int | None = None,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> list[HARDataset]:
    """Χωρίζει το dataset σε per-client subsets για FL εκπαίδευση.

    Strategies:
        by_subject  — ένας client ανά subject (φυσικό Non-IID, default)
        iid         — τυχαίο ισόποσο split (baseline, ιδεατές συνθήκες)
        dirichlet   — ρυθμιζόμενο Non-IID με Dirichlet(alpha)
                      (μικρό alpha → πιο ανισόρροπο → πιο Non-IID)
    """
    X    = dataset.X.numpy()
    y    = dataset.y.numpy()    # 0-indexed εδώ
    subj = dataset.subject_ids
    rng  = np.random.default_rng(seed)

    if strategy == "by_subject":
        n_subjects = int(subj.max()) + 1
        splits = []
        for s in range(n_subjects):
            mask = subj == s
            if mask.sum() == 0:
                continue
            # επαναφορά σε 1-indexed για HARDataset constructor
            client_ds = HARDataset(X[mask], y[mask] + 1)
            client_ds.subject_ids = subj[mask]
            splits.append(client_ds)
        return splits

    if n_clients is None:
        raise ValueError("n_clients απαιτείται για strategies 'iid' και 'dirichlet'.")

    if strategy == "iid":
        # τυχαία ανακατεμένα indices, μοιρασμένα ισόποσα
        indices = rng.permutation(len(y))
        chunks  = np.array_split(indices, n_clients)
        return [HARDataset(X[idx], y[idx] + 1) for idx in chunks]

    if strategy == "dirichlet":
        # Dirichlet split: για κάθε κλάση, μοίρασε τα samples σε clients
        # με αναλογίες που ακολουθούν Dirichlet(alpha)
        # αποτέλεσμα: μικρό alpha → κάποιοι clients έχουν σχεδόν μόνο μία κλάση
        class_indices  = {c: np.where(y == c)[0] for c in np.unique(y)}
        client_buckets: list[list] = [[] for _ in range(n_clients)]

        for c_idx in class_indices.values():
            rng.shuffle(c_idx)
            # proportions: n_clients τιμές που αθροίζουν σε 1
            props   = rng.dirichlet(np.repeat(dirichlet_alpha, n_clients))
            cutoffs = (np.cumsum(props) * len(c_idx)).astype(int)
            cutoffs = np.clip(cutoffs, 0, len(c_idx))
            for cid, part in enumerate(np.split(c_idx, cutoffs[:-1])):
                client_buckets[cid].extend(part.tolist())

        result = []
        for bucket in client_buckets:
            if bucket:
                idx_arr = np.array(bucket)
                result.append(HARDataset(X[idx_arr], y[idx_arr] + 1))
        return result

    raise ValueError(f"Άγνωστο strategy: '{strategy}'")
