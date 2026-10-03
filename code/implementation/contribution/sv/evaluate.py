# Η αξία ενός συνασπισμού, δηλαδή η απόδοση του μοντέλου του στο test.
#
# Ο KernelSHAP ζητά χιλιάδες αξιολογήσεις σε κάθε γύρο. Το test (2.947 x 561) χωρά
# ολόκληρο στη μνήμη, οπότε κάθε αξιολόγηση είναι ένα forward σε όλο το σύνολο,
# χωρίς DataLoader. Το αποτέλεσμα είναι το ίδιο με την test() του train_utils, όπως
# ελέγχει η check_matches_train_utils().
#
# Μετρικές:
#   accuracy     η αξία της εργασίας
#   macro_f1     έλεγχος με άλλη μετρική
#   per_class    διάνυσμα με το recall κάθε δραστηριότητας
#   per_subject  διάνυσμα με το accuracy σε κάθε ένα από τα 9 άτομα του test

import copy
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

METRICS = ("accuracy", "macro_f1", "per_class", "per_subject")


class CoalitionEvaluator:
    def __init__(self, model, dataset, criterion, device, metric="accuracy", groups=None):
        if metric not in METRICS:
            raise ValueError(f"άγνωστη μετρική {metric!r}, διαθέσιμες {METRICS}")
        self.model = copy.deepcopy(model).to(device)
        self.model.eval()
        self.device = device
        self.metric = metric
        self.criterion = criterion
        self.X = dataset.X.to(device)
        self.y = dataset.y.cpu().numpy()
        self._keys = list(self.model.state_dict().keys())
        # Σταθερή σειρά κλάσεων. Χωρίς labels= η sklearn επιστρέφει μόνο όσες
        # προβλέφθηκαν, οπότε το μήκος του διανύσματος θα άλλαζε ανά συνασπισμό.
        self._labels = np.arange(int(self.y.max()) + 1)
        self.n_outputs = len(self._labels) if metric == "per_class" else 1
        # τα βάρη που ανασυνθέτουν το accuracy από τις τιμές του διανύσματος
        self.class_weights = np.array([(self.y == c).sum() for c in self._labels], dtype=np.float64)
        self.class_weights /= self.class_weights.sum()
        self.output_labels = [int(c) for c in self._labels]
        if metric == "per_subject":
            if groups is None or len(groups) != len(self.y):
                raise ValueError("το per_subject χρειάζεται το άτομο κάθε sample του test (groups)")
            uniq, self._group_idx = np.unique(np.asarray(groups), return_inverse=True)
            self._group_counts = np.bincount(self._group_idx).astype(np.float64)
            self.n_outputs = len(uniq)
            self.class_weights = self._group_counts / self._group_counts.sum()
            self.output_labels = [int(u) for u in uniq]

    @torch.no_grad()
    def __call__(self, params: List[np.ndarray]):
        sd = OrderedDict({k: torch.as_tensor(v) for k, v in zip(self._keys, params)})
        self.model.load_state_dict(sd, strict=True)
        pred = self.model(self.X).argmax(1).cpu().numpy()
        if self.metric == "macro_f1":
            return float(f1_score(self.y, pred, average="macro", zero_division=0))
        if self.metric == "per_class":
            return recall_score(self.y, pred, average=None, labels=self._labels,
                                zero_division=0).astype(np.float64)
        if self.metric == "per_subject":
            correct = (pred == self.y).astype(np.float64)
            return np.bincount(self._group_idx, weights=correct,
                               minlength=self.n_outputs) / self._group_counts
        return float(accuracy_score(self.y, pred))

    def check_matches_train_utils(self, params, loader, tol=1e-12) -> float:
        """Η αξιολόγηση εδώ πρέπει να δίνει το ίδιο accuracy με την test() του train_utils."""
        from implementation.common.train_utils import test
        mine = self(params)
        if self.metric != "accuracy":
            return 0.0
        theirs = test(self.model, loader, self.criterion, self.device)[0]
        err = abs(mine - theirs)
        if err > tol:
            raise AssertionError(f"η γρήγορη αξιολόγηση δίνει {mine:.12f}, η test() {theirs:.12f}")
        return err
