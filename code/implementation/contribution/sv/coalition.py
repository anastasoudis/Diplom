# Το συνεργατικό παίγνιο ενός γύρου της ομοσπονδιακής εκπαίδευσης.
#
# Παίκτες είναι οι clients του γύρου. Η αξία v(S) ενός συνασπισμού S είναι η
# απόδοση στο test του μοντέλου που θα έβγαινε στον γύρο, αν είχαν συνεισφέρει μόνο
# οι clients του S. Το v(∅) είναι η απόδοση του global μοντέλου στην αρχή του γύρου.
#
# Το μοντέλο κάθε συνασπισμού δεν εκπαιδεύεται από την αρχή. Ανασυντίθεται από τα
# τοπικά μοντέλα του γύρου με τη στάθμιση του FedAvg, όπως στο GTG-Shapley (Liu κ.ά.
# 2022, σελ. 1 και 8-9). Έτσι κάθε v(S) κοστίζει μία αξιολόγηση και όχι ένα τρέξιμο:
#
#   M_S = Σ_{i∈S} (n_i / Σ_{j∈S} n_j) M_i
#
# Για S = όλοι, το M_S είναι ακριβώς το global μοντέλο του server, αφού υπολογίζεται
# με την ίδια συνάρτηση. Το ελέγχει η assert_matches_fedavg().

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from implementation.federated.fl.aggregation.aggregate import fedavg_aggregate


class RoundGame:
    """
    base          παράμετροι του global μοντέλου στην αρχή του γύρου
    client_params τα τοπικά μοντέλα M_i μετά την εκπαίδευση του γύρου
    sizes         πλήθος samples κάθε client (τα βάρη του FedAvg)
    evaluate      συνάρτηση από παραμέτρους σε αξία
    """

    def __init__(self, base: List[np.ndarray], client_params: List[List[np.ndarray]],
                 sizes: Sequence[int], evaluate: Callable[[List[np.ndarray]], float]):
        if len(client_params) != len(sizes):
            raise ValueError(f"{len(client_params)} μοντέλα αλλά {len(sizes)} μεγέθη")
        self.base = base
        self.client_params = client_params
        # οι ενημερώσεις Δ_i = M_i - M_base, για τον έλεγχο ότι δεν μηδενίστηκαν
        self.deltas = [[p - b for p, b in zip(cp, base)] for cp in client_params]
        self.sizes = np.asarray(sizes, dtype=np.float64)
        self.n = len(client_params)
        self._evaluate = evaluate
        self._cache: Dict[Tuple[int, ...], float] = {}
        self.n_evals = 0

    def params_for(self, mask) -> List[np.ndarray]:
        """Το μοντέλο του συνασπισμού (mask True για όσους ανήκουν σε αυτόν)."""
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if idx.size == 0:
            return [b.copy() for b in self.base]
        return fedavg_aggregate([(self.client_params[i], int(self.sizes[i])) for i in idx])

    def value(self, mask) -> float:
        """v(S). Με την cache κάθε συνασπισμός αξιολογείται μία φορά."""
        key = tuple(int(b) for b in np.asarray(mask, dtype=bool))
        if key not in self._cache:
            self._cache[key] = float(self._evaluate(self.params_for(key)))
            self.n_evals += 1
        return self._cache[key]

    def values(self, masks) -> np.ndarray:
        """Πολλοί συνασπισμοί μαζί. Αυτή τη μορφή καλεί ο shap.KernelExplainer."""
        masks = np.atleast_2d(np.asarray(masks))
        return np.array([self.value(m) for m in masks], dtype=np.float64)

    def empty_value(self):
        return self.value(np.zeros(self.n, dtype=bool))

    def grand_value(self):
        return self.value(np.ones(self.n, dtype=bool))

    def span(self) -> float:
        """v(N) - v(∅), η μεταβολή της απόδοσης σε όλο τον γύρο."""
        return self.grand_value() - self.empty_value()

    def assert_matches_fedavg(self, fedavg_params: List[np.ndarray], tol=0.0) -> float:
        """Η ανασύνθεση του πλήρους συνασπισμού πρέπει να δίνει το μοντέλο του server."""
        mine = self.params_for(np.ones(self.n, dtype=bool))
        err = max(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                  for a, b in zip(mine, fedavg_params))
        if err > tol:
            raise AssertionError(f"η ανασύνθεση διαφέρει από το FedAvg κατά {err:.3e}")
        return err


class ClasswiseRoundGame(RoundGame):
    """Το ίδιο παίγνιο με διανυσματική αξία, μία τιμή ανά δραστηριότητα ή ανά άτομο του test.

    Η ανασύνθεση και η cache είναι ίδιες. Αλλάζει μόνο η συνάρτηση αξίας (Tastan κ.ά.
    2024, Yang κ.ά. 2024). Αν η αξία είναι το recall κάθε δραστηριότητας, τότε
    accuracy = Σ_c w_c recall_c με w_c το ποσοστό της c στο test. Επειδή το Shapley
    είναι γραμμικό ως προς τη συνάρτηση αξίας, ισχύει και φ_i(accuracy) = Σ_c w_c φ_i(recall_c).
    Το ίδιο ισχύει για το accuracy ανά άτομο του test, με βάρη τα ποσοστά των ατόμων.
    """

    def __init__(self, base, client_params, sizes, evaluate, n_outputs: int, class_weights=None):
        super().__init__(base, client_params, sizes, evaluate)
        self.n_outputs = int(n_outputs)
        # τα βάρη w_c, μόνο για την πύλη του γύρου και για τον έλεγχο της ταυτότητας
        self.class_weights = (np.full(self.n_outputs, 1.0 / self.n_outputs) if class_weights is None
                              else np.asarray(class_weights, dtype=np.float64))

    def value(self, mask) -> np.ndarray:
        key = tuple(int(b) for b in np.asarray(mask, dtype=bool))
        if key not in self._cache:
            val = np.asarray(self._evaluate(self.params_for(key)), dtype=np.float64)
            if val.shape != (self.n_outputs,):
                raise ValueError(f"η αξιολόγηση έδωσε σχήμα {val.shape}, αναμενόταν ({self.n_outputs},)")
            self._cache[key] = val
            self.n_evals += 1
        return self._cache[key]

    def values(self, masks) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks))
        return np.stack([self.value(m) for m in masks])

    def span(self) -> float:
        """Το σταθμισμένο άθροισμα των μεταβολών, δηλαδή η μεταβολή του accuracy στον γύρο.

        Έτσι η πύλη του γύρου κρατά τους ίδιους γύρους με το βαθμωτό τρέξιμο.
        """
        return float(self.class_weights @ (self.grand_value() - self.empty_value()))
