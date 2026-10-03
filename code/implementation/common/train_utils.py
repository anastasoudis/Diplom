# Ο βρόχος εκπαίδευσης και ο βρόχος αξιολόγησης. Τους ίδιους καλούν η κεντρική
# εκπαίδευση, οι clients της ομοσπονδιακής και η ατομική.
#
# Δύο διαφορές από τον κώδικα αναφοράς (Simple_FL_Example):
#   1. εκπαιδεύονται ακριβώς `epochs` εποχές (εκεί range(epochs + 1) έδινε μία παραπάνω),
#   2. το accuracy μετριέται μία φορά σε όλο το σύνολο, αφού μαζευτούν όλες οι
#      προβλέψεις. Ο μέσος όρος ανά batch δίνει στο τελευταίο, μικρότερο batch το ίδιο
#      βάρος με τα υπόλοιπα.
#
# Κάθε πέρασμα ενός DataLoader τραβά έναν αριθμό από την καθολική γεννήτρια του torch,
# ακόμη και χωρίς shuffle. Μια αξιολόγηση παραπάνω αλλάζει λοιπόν τη συνέχεια του
# τρεξίματος, οπότε οι αξιολογήσεις γίνονται πάντα στα ίδια σημεία.

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def train(model: nn.Module, trainloader: DataLoader, device, criterion, optimizer,
          epochs: int, print_local: bool = False) -> dict:
    """Εκπαιδεύει το δίκτυο και επιστρέφει accuracy, macro-F1 και loss ανά εποχή."""
    history = {"acc": [], "loss": [], "f1": []}
    for ep in range(epochs):
        model.train()
        running, nb = None, 0
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            # άθροιση πάνω στη συσκευή, χωρίς .item() σε κάθε batch
            running = loss.detach() if running is None else running + loss.detach()
            nb += 1
        acc, f1 = test(model, trainloader, criterion, device)[:2]
        history["acc"].append(acc)
        history["f1"].append(f1)
        history["loss"].append(float(running) / max(nb, 1) if nb else 0.0)
        if print_local:
            print(f"    εποχή {ep + 1:>2}/{epochs}, loss {history['loss'][-1]:.4f}, "
                  f"acc {acc:.4f}, f1 {f1:.4f}")
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, criterion, device):
    """(y_true, y_pred, μέση απώλεια ανά batch) σε όλο το σύνολο του loader."""
    if len(loader.dataset) == 0:
        raise ValueError("άδειο σύνολο δεδομένων")
    model.eval()
    ys, ps, tot, nb = [], [], None, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb).detach()
        tot = loss if tot is None else tot + loss
        nb += 1
        ys.append(yb)
        ps.append(out.argmax(1))
    y = torch.cat(ys).cpu().numpy()
    p = torch.cat(ps).cpu().numpy()
    return y, p, (float(tot) / max(nb, 1) if nb else 0.0)


@torch.no_grad()
def test(model: nn.Module, testloader: DataLoader, criterion, device) -> Tuple[float, float, float]:
    """(accuracy, macro-F1, loss) σε όλο το σύνολο."""
    y, p, loss = predict(model, testloader, criterion, device)
    return (float(accuracy_score(y, p)),
            float(f1_score(y, p, average="macro", zero_division=0)),
            loss)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
