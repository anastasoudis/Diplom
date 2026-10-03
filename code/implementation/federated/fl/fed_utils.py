# Δημιουργία και αρχικοποίηση των clients.
#
# Ένας client για κάθε εθελοντή του train, με τα δικά του samples. Η κατανομή των
# δεδομένων στους clients είναι αυτή που υπάρχει στο dataset και δεν κατασκευάζεται.

import copy

import numpy as np
from torch.utils.data import Subset

from implementation.federated.fl.client import Client


def create_fed_clients(trainset, client_ids):
    clients = []
    for cid in range(int(np.max(client_ids)) + 1):
        idx = np.where(client_ids == cid)[0]
        if len(idx):
            clients.append(Client(cid, Subset(trainset, idx.tolist())))
    return clients


def initialize_fed_clients(client_list, args, model):
    params = {"epochs": args.epochs, "lr": args.lr, "device": args.device,
              "batch_size": args.batch_size, "criterion": args.criterion,
              "optimizer": args.optimizer}
    for cl in client_list:
        cl.init_parameters(params, copy.deepcopy(model))
    return client_list
