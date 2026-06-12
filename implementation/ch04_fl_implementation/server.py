# =============================================================================
# ch04_fl_implementation / server.py
# -----------------------------------------------------------------------------
# FL Server: συντονισμός γύρων εκπαίδευσης, FedAvg aggregation,
# και παρακολούθηση συνεισφοράς clients (για το incentive module).
#
# Αντιστοιχεί σε: §4.2 (Federated Framework) της διπλωματικής
# =============================================================================

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from implementation.ch04_fl_implementation.model import HAR_CNN, get_model, count_parameters
from implementation.ch04_fl_implementation.client import FLClient


def compute_classification_metrics(y_true, y_pred) -> Dict:
    """Πλήρεις μετρικές ταξινόμησης από y_true/y_pred.

    Επιστρέφει accuracy, F1-macro, F1-weighted, per-class report (precision/
    recall/F1/support) και confusion matrix. Χρησιμοποιείται για τα τελικά
    αποτελέσματα της διπλωματικής (το per-round tracking μένει μόνο accuracy).
    """
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy":         float((y_true == y_pred).mean()),
        "f1_macro":         float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted":      float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class":        classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples":        int(len(y_true)),
    }


# Παράμετροι FL εκπαίδευσης
FL_ROUNDS         = 20    # συνολικοί γύροι επικοινωνίας
MIN_CLIENTS       = 2     # ελάχιστος αριθμός clients ανά γύρο
FRACTION_FIT      = 1.0   # ποσοστό clients που συμμετέχουν ανά γύρο (1.0 = όλοι)


class FLServer:
    """Συντονιστής του Federated Learning συστήματος.

    Ο server δεν έχει ποτέ πρόσβαση στα raw δεδομένα των clients.
    Λαμβάνει μόνο παραμέτρους (weights) και αριθμό δειγμάτων.

    Αρμοδιότητες:
        - Αρχικοποίηση global model
        - Επιλογή clients ανά γύρο
        - FedAvg aggregation των τοπικών παραμέτρων
        - Παρακολούθηση history (loss, accuracy) ανά γύρο
        - Αποθήκευση n_samples ανά client για LOO contribution (§4.3)
    """

    def __init__(self, clients: List[FLClient], device: str = "cpu"):
        self.clients = clients
        self.device  = device

        # το global model αρχικοποιείται με τυχαίες παραμέτρους (Xavier init από PyTorch)
        self.model: HAR_CNN = get_model(device)

        # ιστορικό ανά γύρο — γράφεται από run_round(), διαβάζεται από evaluate_global()
        self.history: Dict[str, List] = defaultdict(list)

        # αριθμός δειγμάτων ανά client — απαιτείται για FedAvg και LOO
        self.client_n_samples: Dict[int, int] = {}

        print(f"FLServer initialized: {len(clients)} clients, "
              f"{count_parameters(self.model):,} parameters")

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def fedavg(self,
               param_list:  List[List[np.ndarray]],
               weight_list: List[int]
               ) -> List[np.ndarray]:
        """Σταθμισμένος μέσος παραμέτρων (FedAvg, McMahan et al. 2017).

        Κάθε client συνεισφέρει ανάλογα με τον αριθμό των δειγμάτων του:
            θ_global = Σ_i (n_i / N) · θ_i
        όπου N = Σ_i n_i (συνολικά δείγματα όλων των clients).

        Args:
            param_list:  λίστα από λίστες numpy arrays (ένα per client)
            weight_list: n_samples κάθε client — οι βάρη της στάθμισης

        Returns:
            Νέες global παράμετροι ως λίστα numpy arrays.
        """
        total_samples = sum(weight_list)

        # αρχικοποίηση με float64 μηδενικά — αναγκαίο γιατί κάποιες παράμετροι
        # (π.χ. BatchNorm.num_batches_tracked) είναι int64 εσωτερικά
        aggregated = [np.zeros_like(p, dtype=np.float64) for p in param_list[0]]

        for params, n in zip(param_list, weight_list):
            weight = n / total_samples   # κλάσμα συνεισφοράς αυτού του client
            for i, param in enumerate(params):
                aggregated[i] += weight * param

        return aggregated

    # -------------------------------------------------------------------------
    # Ένας FL γύρος
    # -------------------------------------------------------------------------

    def run_round(self, round_num: int,
                  fraction: float = FRACTION_FIT
                  ) -> Dict[str, float]:
        """Εκτελεί έναν γύρο FL: διανομή → τοπική εκπαίδευση → aggregation.

        Ροή:
            1. Επιλογή clients (random subset με αναλογία `fraction`)
            2. Αποστολή global params σε κάθε επιλεγμένο client
            3. Κάθε client εκπαιδεύει τοπικά (LOCAL_EPOCHS εποχές)
            4. Συλλογή ενημερωμένων params + n_samples
            5. FedAvg aggregation → νέο global model
            6. Επιστροφή μετρικών γύρου

        Returns:
            dict με 'loss' και 'accuracy' (σταθμισμένος μέσος κλάσεων)
        """
        global_params = [v.cpu().numpy() for v in self.model.state_dict().values()]

        # επιλογή clients για αυτόν τον γύρο
        n_selected = max(MIN_CLIENTS, int(len(self.clients) * fraction))
        n_selected = min(n_selected, len(self.clients))   # cap: όχι πάνω από τους διαθέσιμους (π.χ. subsets 1 client στο Shapley)
        rng = np.random.default_rng(seed=round_num)   # reproducible per round
        selected = rng.choice(len(self.clients), size=n_selected, replace=False)
        selected_clients = [self.clients[i] for i in selected]

        # συλλογή αποτελεσμάτων τοπικής εκπαίδευσης
        all_params:   List[List[np.ndarray]] = []
        all_n:        List[int]              = []
        round_losses: List[float]            = []
        round_accs:   List[float]            = []

        for client in selected_clients:
            params, n_samples, metrics = client.fit(self.model, global_params)

            all_params.append(params)
            all_n.append(n_samples)
            round_losses.append(metrics["loss"])
            round_accs.append(metrics["accuracy"])

            # αποθήκευση n_samples — χρησιμοποιείται από incentive.py για LOO
            self.client_n_samples[client.id] = n_samples

        # FedAvg: νέο global model
        new_global_params = self.fedavg(all_params, all_n)
        self._load_params(new_global_params)

        # σταθμισμένος μέσος μετρικών (ανάλογα με n_samples)
        total = sum(all_n)
        avg_loss = sum(l * n for l, n in zip(round_losses, all_n)) / total
        avg_acc  = sum(a * n for a, n in zip(round_accs,   all_n)) / total

        round_metrics = {"loss": avg_loss, "accuracy": avg_acc}
        self.history["loss"].append(avg_loss)
        self.history["accuracy"].append(avg_acc)

        print(f"  Round {round_num:2d}/{FL_ROUNDS}: "
              f"loss={avg_loss:.4f}, acc={avg_acc:.4f} "
              f"(clients={n_selected}/{len(self.clients)})")

        return round_metrics

    # -------------------------------------------------------------------------
    # Αξιολόγηση global model
    # -------------------------------------------------------------------------

    def evaluate_global(self, test_clients: Optional[List[FLClient]] = None
                        ) -> Tuple[float, float]:
        """Αξιολογεί το global model σε όλους τους clients (ή σε δοσμένο σύνολο).

        Χρησιμοποιείται στο τέλος κάθε γύρου για αξιολόγηση γενίκευσης.
        Επιστρέφει (loss, accuracy) σταθμισμένο ανά client.
        """
        clients_to_eval = test_clients if test_clients is not None else self.clients
        global_params   = [v.cpu().numpy() for v in self.model.state_dict().values()]

        total_loss    = 0.0
        total_correct = 0.0
        total_samples = 0

        for client in clients_to_eval:
            loss, acc = client.evaluate(global_params)
            n = client.n_samples
            total_loss    += loss * n
            total_correct += acc  * n
            total_samples += n

        return total_loss / total_samples, total_correct / total_samples

    def evaluate_global_detailed(self, test_clients: Optional[List[FLClient]] = None
                                 ) -> Dict:
        """Πλήρης αξιολόγηση του τελικού global model (accuracy, F1-macro,
        per-class F1, confusion matrix) σε δοσμένο σύνολο αξιολόγησης.

        Σε αντίθεση με την evaluate_global (που επιστρέφει μόνο σταθμισμένο
        accuracy ανά γύρο), εδώ μαζεύονται ΟΛΕΣ οι προβλέψεις ώστε να
        υπολογιστούν global μετρικές. Καλείται μία φορά στο τέλος.
        """
        import torch
        clients_to_eval = test_clients if test_clients is not None else self.clients
        self.model.eval()
        self.model.to(self.device)

        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for client in clients_to_eval:
                for X_batch, y_batch in client.loader:
                    logits = self.model(X_batch.to(self.device))
                    y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                    y_true.extend(y_batch.numpy().tolist())

        return compute_classification_metrics(y_true, y_pred)

    # -------------------------------------------------------------------------
    # Βοηθητικές μέθοδοι
    # -------------------------------------------------------------------------

    def _load_params(self, params: List[np.ndarray]) -> None:
        """Φορτώνει λίστα numpy arrays στο global model (state_dict).

        Κάνει cast στο αρχικό dtype κάθε παραμέτρου ώστε τα int64 buffers
        (π.χ. BatchNorm.num_batches_tracked) να παραμείνουν ακέραια.
        """
        from collections import OrderedDict
        import torch
        state_keys    = list(self.model.state_dict().keys())
        original_dtypes = {k: v.dtype
                           for k, v in self.model.state_dict().items()}
        state_dict = OrderedDict()
        for k, v in zip(state_keys, params):
            tensor = torch.tensor(v).to(original_dtypes[k])
            state_dict[k] = tensor
        self.model.load_state_dict(state_dict, strict=True)

    def get_global_params(self) -> List[np.ndarray]:
        """Επιστρέφει τις τρέχουσες global παραμέτρους."""
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def run(self, n_rounds: int = FL_ROUNDS) -> Dict[str, List]:
        """Εκτελεί n_rounds γύρους FL και επιστρέφει το ιστορικό εκπαίδευσης.

        Αυτή είναι η κεντρική μέθοδος που καλείται από το experiment script.
        """
        print(f"\nStarting FL training: {n_rounds} rounds, "
              f"{len(self.clients)} clients\n" + "-" * 55)

        for r in range(1, n_rounds + 1):
            self.run_round(r)

        print("-" * 55)
        final_loss, final_acc = self.evaluate_global()
        print(f"Final global model: loss={final_loss:.4f}, acc={final_acc:.4f}")

        return dict(self.history)
