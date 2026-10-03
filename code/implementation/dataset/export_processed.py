# Εξαγωγή των επεξεργασμένων δεδομένων στο data/har_processed και φόρτωσή τους για τα πειράματα.
#
# Η εξαγωγή διαβάζει τα ωμά αρχεία, εφαρμόζει τη z-score του load_uci_har() και
# γράφει CSV. Το PROVENANCE.json κρατά το sha256 των ωμών αρχείων και των εξόδων.
# Η load_processed() ελέγχει το sha256 κάθε αρχείου πριν επιστρέψει δεδομένα, ώστε
# κάθε τρέξιμο να γίνεται στα ίδια ακριβώς δεδομένα.
#
#   python -m implementation.dataset.export_processed           # εξαγωγή
#   python -m implementation.dataset.export_processed --verify  # έλεγχος ωμών και εξόδων

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from implementation.dataset.load_dataset import ROOT, UCI_DIR, HARDataset, load_uci_har

OUT_DIR = os.path.join(ROOT, "data", "har_processed")
SRC_FILES = ["train/X_train.txt", "train/y_train.txt", "train/subject_train.txt",
             "test/X_test.txt", "test/y_test.txt", "test/subject_test.txt",
             "features.txt", "activity_labels.txt"]
# τα πεδία του PROVENANCE.json που αντιγράφει κάθε αποτέλεσμα ως data_provenance
PROVENANCE_KEYS = ("pipeline", "source_sha256", "output_sha256", "created")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def feature_names(src_dir=UCI_DIR):
    """Τα 561 ονόματα χαρακτηριστικών, μαζί με τις στήλες που επαναλαμβάνουν άλλες.

    Στο features.txt 42 ονόματα εμφανίζονται τρεις φορές (τα bandsEnergy() των
    fBodyAcc, fBodyAccJerk, fBodyGyro). Οι τρεις εμφανίσεις είναι οι άξονες X, Y, Z,
    με διαφορετικές τιμές, οπότε εδώ παίρνουν κατάληξη -X, -Y, -Z. Επιπλέον 21 στήλες
    είναι ίδιες με κάποια προηγούμενη. Το duplicate_of δίνει ποια (από 1, 0 αν είναι μοναδική).
    """
    raw = [l.split(None, 1)[1].strip()
           for l in open(os.path.join(src_dir, "features.txt"), encoding="utf-8")]
    seen, names = {}, []
    for n in raw:
        k = seen[n] = seen.get(n, -1) + 1
        names.append(f"{n}-{'XYZ'[k]}" if raw.count(n) == 3 else n)

    X = np.loadtxt(os.path.join(src_dir, "train", "X_train.txt"), dtype=np.float64)
    first, dup_of = {}, [0] * len(names)
    for j in range(X.shape[1]):
        key = X[:, j].tobytes()
        if key in first:
            dup_of[j] = first[key] + 1
        else:
            first[key] = j
    return names, raw, dup_of


def build():
    """Οι πίνακες που γράφονται και τα μεταδεδομένα τους."""
    trainset, testset, meta = load_uci_har()
    s_test = np.loadtxt(os.path.join(UCI_DIR, "test", "subject_test.txt"), dtype=np.int32)
    # Τα άτομα του test κρατούν τους αρχικούς αριθμούς του UCI, ώστε να μη
    # συγχέονται με τους clients 0-20.
    meta["test_subjects"] = sorted(set(s_test.tolist()))
    arrays = {"X_train.csv": trainset.X.numpy().reshape(len(trainset), -1),
              "y_train.csv": (trainset.y.numpy() + 1).astype(np.int32),
              "X_test.csv": testset.X.numpy().reshape(len(testset), -1),
              "y_test.csv": (testset.y.numpy() + 1).astype(np.int32),
              "client_ids_subject.csv": trainset.subject_ids.astype(np.int32),
              "subject_ids_test.csv": s_test}
    return arrays, meta


def write(arrays, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, a in arrays.items():
        fmt = "%.6f" if a.dtype.kind == "f" else "%d"
        np.savetxt(os.path.join(out_dir, name), a, delimiter=",", fmt=fmt)
    names, raw, dup_of = feature_names()
    # με csv.writer, γιατί ονόματα όπως «arCoeff()-X,1» περιέχουν κόμμα
    with open(os.path.join(out_dir, "feature_names.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["index", "name", "name_raw", "duplicate_of"])
        for i, (n, r, d) in enumerate(zip(names, raw, dup_of), start=1):
            w.writerow([i, n, r, d or ""])
    return {n: sha256(os.path.join(out_dir, n)) for n in list(arrays) + ["feature_names.csv"]}


def verify(out_dir):
    prov_path = os.path.join(out_dir, "PROVENANCE.json")
    if not os.path.exists(prov_path):
        print(f"δεν υπάρχει το {prov_path}, τρέξε πρώτα την εξαγωγή")
        return 1
    prov = json.load(open(prov_path, encoding="utf-8"))
    ok = True
    for kind, base, hashes in (("πηγή", UCI_DIR, prov["source_sha256"]),
                               ("έξοδος", out_dir, prov["output_sha256"])):
        for n, h in hashes.items():
            p = os.path.join(base, n)
            same = os.path.exists(p) and sha256(p) == h
            ok &= same
            print(f"  {kind:<7}{n:<32}{'ίδιο' if same else 'διαφέρει'}")
    print("τα αρχεία συμφωνούν με το PROVENANCE.json" if ok else "υπάρχουν διαφορές, ξανατρέξε την εξαγωγή")
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--verify" in argv:
        return verify(OUT_DIR)

    arrays, meta = build()
    out = write(arrays, OUT_DIR)
    prov = {"dataset": "UCI HAR",
            "pipeline": "ids εθελοντών όπως δίνονται, μέσος και τυπική απόκλιση από το train, z-score",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": "python -m implementation.dataset.export_processed",
            "source_sha256": {f: sha256(os.path.join(UCI_DIR, f)) for f in SRC_FILES},
            "output_sha256": out,
            "shapes": {n: list(a.shape) for n, a in arrays.items()},
            "meta": meta}
    with open(os.path.join(OUT_DIR, "PROVENANCE.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print("εξαγωγή στο", os.path.relpath(OUT_DIR, ROOT))
    for n, h in out.items():
        print(f"  {n:<24} sha256 {h}")
    return 0


def load_processed(out_dir=OUT_DIR):
    """(trainset, testset, client_ids, provenance) από το data/har_processed.

    Πριν διαβάσει οτιδήποτε ελέγχει το sha256 κάθε αρχείου απέναντι στο
    PROVENANCE.json. client_ids[i] είναι ο client (0-20) του i-οστού sample του train.
    """
    prov = json.load(open(os.path.join(out_dir, "PROVENANCE.json"), encoding="utf-8"))
    bad = [n for n, h in prov["output_sha256"].items()
           if not os.path.exists(os.path.join(out_dir, n)) or sha256(os.path.join(out_dir, n)) != h]
    if bad:
        raise RuntimeError(f"τα αρχεία δεν ταιριάζουν με το PROVENANCE.json: {bad}")

    load = lambda n: np.loadtxt(os.path.join(out_dir, n), delimiter=",")
    X_train, X_test = load("X_train.csv"), load("X_test.csv")
    y_train, y_test = load("y_train.csv").astype(int), load("y_test.csv").astype(int)
    client_ids = load("client_ids_subject.csv").astype(int)
    test_subjects = load("subject_ids_test.csv").astype(int)
    trainset = HARDataset(X_train[:, None, :], y_train, subject_ids=client_ids)
    testset = HARDataset(X_test[:, None, :], y_test, subject_ids=test_subjects)
    return trainset, testset, client_ids, prov


def provenance_block(prov):
    return {k: prov[k] for k in PROVENANCE_KEYS}


def common_provenance(blocks):
    """Ένα data_provenance για αποτέλεσμα που ενώνει πολλά μέρη.

    Σταματά αν δύο μέρη έτρεξαν σε διαφορετικά δεδομένα. Συγκρίνονται μόνο τα
    αποτυπώματα, γιατί η ημερομηνία αλλάζει και σε ταυτόσημη επανεξαγωγή.
    """
    if not blocks or any(b is None for b in blocks):
        return None
    fps = {json.dumps([b["source_sha256"], b["output_sha256"]], sort_keys=True) for b in blocks}
    if len(fps) > 1:
        raise ValueError("τα μέρη έτρεξαν σε διαφορετικά δεδομένα")
    return blocks[0]


if __name__ == "__main__":
    sys.exit(main())
