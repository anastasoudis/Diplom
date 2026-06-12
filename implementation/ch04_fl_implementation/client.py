# =============================================================================
# ch04_fl_implementation / client.py
# -----------------------------------------------------------------------------
# FL Client: εκπαίδευση τοπικού μοντέλου και επικοινωνία με τον server.
# Κάθε client αντιστοιχεί σε έναν χρήστη (subject) του AAL dataset.
#
# Αντιστοιχεί σε: §4.2 (Federated Framework) της διπλωματικής
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Tuple

from implementation.ch04_fl_implementation.model import HAR_CNN
from implementation.ch03_data_analysis.dataset_loader import HARDataset


# Παράμετροι τοπικής εκπαίδευσης
LOCAL_EPOCHS = 5       # εποχές ανά FL γύρο — trade-off: περισσότερες → client drift
BATCH_SIZE   = 32
LEARNING_RATE = 1e-3


class FLClient:
    """Αναπαριστά έναν συμμετέχοντα (client) στο Federated Learning σύστημα.

    Ο client κρατά τοπικά δεδομένα, εκπαιδεύει το μοντέλο τοπικά, και
    επιστρέφει μόνο τις παραμέτρους — ποτέ τα raw δεδομένα.
    """

    def __init__(self, client_id: int, dataset: HARDataset, device: str = "cpu"):
        self.id      = client_id
        self.dataset = dataset
        self.device  = device
        self.n_samples = len(dataset)   # χρησιμοποιείται για σταθμισμένο FedAvg

        self.loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 drop_last=False)

        # το μοντέλο αρχικοποιείται όταν ο server στείλει τις πρώτες παραμέτρους
        self.model: HAR_CNN | None = None

    def set_parameters(self, model: HAR_CNN, params: List[np.ndarray]) -> None:
        """Φορτώνει παραμέτρους από τον server στο τοπικό μοντέλο.

        Η αντιστοίχηση γίνεται με το state_dict — κάθε key αντιστοιχεί
        σε ένα layer (π.χ. 'conv1.0.weight', 'classifier.1.bias', ...).
        """
        self.model = model
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """Επιστρέφει τις τοπικές παραμέτρους ως λίστα από numpy arrays.

        Ο server δέχεται αυτή τη μορφή για να κάνει FedAvg aggregation.
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, model: HAR_CNN, global_params: List[np.ndarray]
            ) -> Tuple[List[np.ndarray], int, dict]:
        """Εκπαιδεύει τοπικά για LOCAL_EPOCHS εποχές και επιστρέφει νέες παραμέτρους.

        Αυτό είναι το κεντρικό βήμα του FL: ο client παίρνει το global model,
        το βελτιώνει με τα δικά του δεδομένα, και στέλνει πίσω μόνο τα weights.

        Returns:
            (updated_params, n_samples, metrics)
        """
        self.set_parameters(model, global_params)
        self.model.train()
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        total_loss   = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(LOCAL_EPOCHS):
            for X_batch, y_batch in self.loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss    += loss.item() * len(y_batch)
                preds          = logits.argmax(dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += len(y_batch)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        metrics = {"loss": avg_loss, "accuracy": accuracy}
        return self.get_parameters(), self.n_samples, metrics

    def evaluate(self, params: List[np.ndarray]) -> Tuple[float, float]:
        """Αξιολογεί το μοντέλο με τις δεδομένες παραμέτρους στα τοπικά δεδομένα.

        Χρησιμοποιείται για υπολογισμό συνεισφοράς (LOO) στον server.
        Επιστρέφει (loss, accuracy).
        """
        if self.model is None:
            # evaluation-only client (π.χ. held-out test set) που δεν πέρασε από fit()
            from implementation.ch04_fl_implementation.model import get_model
            self.model = get_model(self.device)
        self.set_parameters(self.model, params)
        self.model.eval()
        self.model.to(self.device)

        criterion     = nn.CrossEntropyLoss()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in self.loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits  = self.model(X_batch)
                loss    = criterion(logits, y_batch)

                total_loss    += loss.item() * len(y_batch)
                preds          = logits.argmax(dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += len(y_batch)

        return total_loss / total_samples, total_correct / total_samples
