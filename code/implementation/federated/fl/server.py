# Ο server της ομοσπονδιακής εκπαίδευσης. Σε κάθε γύρο επιλέγει τους clients, τους αφήνει
# να εκπαιδεύσουν τοπικά, κάνει τη συνάθροιση FedAvg και αξιολογεί το global μοντέλο στο test.

from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from implementation.common.helpers import get_criterion
from implementation.common.metrics import compute_metrics
from implementation.common.train_utils import predict, test
from implementation.federated.fl.aggregation.aggregate import fedavg_aggregate
from implementation.federated.fl.selectors import get_selector


class Server:
    def __init__(self, args, testset, model):
        self.testloader = DataLoader(testset, batch_size=256)
        self.model = model
        self.device = args.device
        self.criterion = get_criterion(args.criterion)
        self.selector = get_selector(getattr(args, "selector", "all"), getattr(args, "exclude", None))

    def update(self, client_list, observer=None):
        """Ένας γύρος. Επιστρέφει τους clients με το νέο global μοντέλο.

        Αν δοθεί observer, καλείται αφού εκπαιδευτούν οι clients και πριν από τη
        συνάθροιση, με τις παραμέτρους του global στην αρχή του γύρου και τους
        επιλεγμένους clients. Από εκεί η αποτίμηση συνεισφοράς παίρνει τα τοπικά
        μοντέλα του γύρου, χωρίς δεύτερο αντίγραφο του βρόχου.
        """
        selected = self.selector.sample_clients(client_list)
        for cl in selected:
            cl.update()
        if observer is not None:
            observer(self.get_server_parameters(), selected)
        return self.perform_federated_aggregation(client_list, selected)

    def evaluate(self):
        """(accuracy, macro-F1, loss) του global μοντέλου στο test."""
        return test(self.model, self.testloader, self.criterion, self.device)

    def evaluate_full(self):
        """Όλες οι μετρικές, μαζί με τον πίνακα σύγχυσης."""
        y, p, loss = predict(self.model, self.testloader, self.criterion, self.device)
        return compute_metrics(y, p, loss)

    def set_server_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if isinstance(parameters, torch.nn.Module):
            self.model.load_state_dict(parameters.state_dict(), strict=True)
        else:
            sd = OrderedDict({k: torch.as_tensor(v) for k, v in
                              zip(self.model.state_dict().keys(), parameters)})
            self.model.load_state_dict(sd, strict=True)

    def get_server_parameters(self) -> List[np.ndarray]:
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def perform_federated_aggregation(self, client_list, selected_clients):
        results: List[Tuple[List[np.ndarray], int]] = [
            (cl.get_parameters(), len(cl.dataset)) for cl in selected_clients]
        aggregated = fedavg_aggregate(results)
        self.set_server_parameters(aggregated)
        for cl in client_list:
            cl.set_parameters(aggregated)
        return client_list
