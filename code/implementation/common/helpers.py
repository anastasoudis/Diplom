# Optimizer, συνάρτηση απώλειας και συσκευή από το όνομά τους.

import torch


def get_optim(model, optim_name: str = "adam", lr: float = 1e-3):
    if optim_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optim_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise NotImplementedError(f"άγνωστος optimizer {optim_name!r}")


def get_criterion(crit_name: str = "cross_entropy"):
    if crit_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    raise NotImplementedError(f"άγνωστη συνάρτηση απώλειας {crit_name!r}")


def resolve_device(name: str) -> str:
    """Η συσκευή που ζητήθηκε, ή σφάλμα αν δεν υπάρχει.

    Δεν πέφτει σιωπηλά στη CPU, γιατί τότε τα args θα έγραφαν άλλη συσκευή από
    αυτήν που έτρεξε. Τα αποτελέσματα της εργασίας βγήκαν όλα σε CPU.
    """
    if name == "cpu":
        return "cpu"
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(f"η συσκευή {name!r} δεν είναι διαθέσιμη")
