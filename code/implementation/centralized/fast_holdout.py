# Η εκπαίδευση του run_holdout_one χωρίς DataLoader, για το παίγνιο επανεκπαίδευσης.
#
# Το Shapley με παίκτες τους 21 εθελοντές στην κεντρική εκπαίδευση θέλει μία νέα
# εκπαίδευση για κάθε συνασπισμό, δηλαδή δεκάδες χιλιάδες εκπαιδεύσεις. Εδώ τα
# batches κόβονται απευθείας από τους τανυστές και δεν μετριέται το accuracy στο
# train σε κάθε εποχή, που το παίγνιο δεν χρειάζεται.
#
# Το αποτέλεσμα πρέπει να είναι το ίδιο μοντέλο με του run_holdout_one, αλλιώς το
# v(N) του παιγνίου δεν θα ήταν το μοντέλο της κεντρικής εκπαίδευσης. Γι' αυτό
# αναπαράγονται και οι κλήσεις στην καθολική γεννήτρια του torch που κάνει ο
# κανονικός βρόχος (torch 2.11):
#   κάθε πέρασμα ενός DataLoader              1 αριθμός
#   RandomSampler, στην αρχή κάθε εποχής      1 αριθμός (seed του randperm)
#   test(model, tr) στο τέλος κάθε εποχής     2
#   σε εποχή του --checkpoints                1 (predict στο test) και 2 (test στο train)
# Οι αξιολογήσεις γίνονται σε eval(), χωρίς dropout, οπότε μόνο οι κλήσεις αυτές
# επηρεάζουν τη συνέχεια. Η verify_fast_holdout() συγκρίνει τα δύο.

import torch

from implementation.centralized.centralized import build
from implementation.common.train_utils import set_seed


def _draw(k: int = 1) -> int:
    """k αριθμοί int64 από την καθολική γεννήτρια, όπως ο DataLoader. Επιστρέφει τον τελευταίο."""
    v = 0
    for _ in range(k):
        v = int(torch.empty((), dtype=torch.int64).random_().item())
    return v


def fast_holdout_predict(args, X, y, Xte, seed, device="cpu"):
    """Οι προβλέψεις στο test του μοντέλου που θα έβγαζε το run_holdout_one.

    X (N, 1, 561) float32, y (N,) με ετικέτες 0-5, Xte (M, 1, 561), με τη σειρά
    samples που θα είχε το dataset του DataLoader.
    """
    set_seed(seed)
    model, crit, opt = build(args, device)
    marks = set([e for e in args.checkpoints if e <= args.epochs] + [args.epochs])
    n, bs = len(y), args.batch_size
    X, y = X.to(device), y.to(device)

    for ep in range(1, args.epochs + 1):
        model.train()
        _draw(1)                                    # πέρασμα του DataLoader
        g = torch.Generator()
        g.manual_seed(_draw(1))                     # seed του RandomSampler
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
        _draw(2)                                    # test(model, tr) στο τέλος του train()
        if ep in marks:
            _draw(1)                                # predict στο test
            _draw(2)                                # test στο train

    model.eval()
    with torch.no_grad():
        return model(Xte.to(device)).argmax(1).cpu().numpy()


def verify_fast_holdout(args, trainset, testset, seed, device="cpu"):
    """Ίδιος πίνακας σύγχυσης και accuracy με το run_holdout_one;"""
    from implementation.centralized.centralized import run_holdout_one
    from implementation.common.metrics import compute_metrics
    slow = run_holdout_one(args, trainset, testset, device, seed)
    fast = compute_metrics(testset.y.numpy(),
                           fast_holdout_predict(args, trainset.X, trainset.y, testset.X, seed, device))
    same = fast["accuracy"] == slow["accuracy"] and fast["confusion_matrix"] == slow["confusion_matrix"]
    return same, slow["accuracy"], fast["accuracy"]
