# Ένας client της ομοσπονδιακής εκπαίδευσης. Ίδιες μέθοδοι με τον κώδικα αναφοράς
# (init_parameters, set_parameters, get_parameters, update).
#
# Κάθε client παίρνει δικό του αντίγραφο του μοντέλου. Στον κώδικα αναφοράς το
# deepcopy γίνεται μία φορά έξω από τον βρόχο, οπότε όλοι οι clients μοιράζονται το
# ίδιο αντικείμενο και η εκπαίδευση γίνεται στην πράξη σειριακά.

import copy
from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from implementation.common.helpers import get_criterion, get_optim
from implementation.common.train_utils import train


class Client:
    def __init__(self, cid, dataset):
        self.id = cid
        self.dataset = dataset
        self.model = None

    def init_parameters(self, params: Dict, model):
        self.epochs = params["epochs"]
        self.device = params["device"]
        self.model = copy.deepcopy(model)
        self.criterion = get_criterion(params["criterion"])
        self.optimizer = get_optim(self.model, params["optimizer"], params["lr"])
        # όλα τα samples του client εκπαιδεύουν, όπως στην κεντρική εκπαίδευση
        self.train_loader = DataLoader(self.dataset, batch_size=params["batch_size"], shuffle=True)

    def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if isinstance(parameters, torch.nn.Module):
            self.model.load_state_dict(parameters.state_dict(), strict=True)
        else:
            sd = OrderedDict({k: torch.as_tensor(v) for k, v in
                              zip(self.model.state_dict().keys(), parameters)})
            self.model.load_state_dict(sd, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        # Για τανυστές στη CPU το .numpy() δίνει όψη στην ίδια μνήμη. Όποιος θέλει να
        # κρατήσει τις τιμές μετά τη συνάθροιση πρέπει να τις αντιγράψει.
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def update(self):
        return train(self.model, self.train_loader, self.device,
                     self.criterion, self.optimizer, self.epochs)
