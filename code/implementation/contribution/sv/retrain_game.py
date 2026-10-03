# Το παίγνιο επανεκπαίδευσης, για Shapley στην κεντρική εκπαίδευση με παίκτες τους 21 εθελοντές.
#
# v(S) είναι το accuracy στο test ενός δικτύου που εκπαιδεύεται από την αρχή μόνο με τα
# samples των εθελοντών του S, με τις ρυθμίσεις της κεντρικής εκπαίδευσης. Σε αντίθεση
# με το παίγνιο του coalition.py, εδώ κάθε συνασπισμός είναι νέα εκπαίδευση. Είναι το
# Data Shapley των Ghorbani και Zou (2019), με πηγή δεδομένων τα samples ενός ατόμου.
#
# Όλοι οι συνασπισμοί ενός seed εκπαιδεύονται με το ίδιο seed, οπότε το v(N) είναι το
# μοντέλο της κεντρικής εκπαίδευσης αυτού του seed. Το v(∅) είναι το accuracy του
# δικτύου όπως αρχικοποιείται με το seed, χωρίς εκπαίδευση. Οι νέοι συνασπισμοί
# μοιράζονται σε διεργασίες με ένα νήμα η καθεμία.

from typing import Dict, Tuple

import numpy as np

_W: Dict = {}   # τα δεδομένα κάθε διεργασίας, φορτωμένα μία φορά


def init_worker(args):
    import torch
    torch.set_num_threads(1)
    from implementation.dataset.export_processed import load_processed
    tr, te, client_ids, _ = load_processed()
    _W.update(args=args, X=tr.X, y=tr.y, Xte=te.X, yte=te.y.numpy(), cids=np.asarray(client_ids))


def _accuracy_for(seed: int, members: Tuple[int, ...]) -> float:
    import torch
    from implementation.centralized.centralized import build
    from implementation.centralized.fast_holdout import fast_holdout_predict
    from implementation.common.train_utils import set_seed
    a = _W["args"]
    if not members:
        set_seed(seed)
        model, _, _ = build(a, "cpu")
        model.eval()
        with torch.no_grad():
            pred = model(_W["Xte"]).argmax(1).numpy()
    else:
        # τα samples με αύξουσα σειρά, όπως θα τα έδινε ένα Subset
        idx = np.where(np.isin(_W["cids"], members))[0]
        pred = fast_holdout_predict(a, _W["X"][idx], _W["y"][idx], _W["Xte"], seed)
    return float((pred == _W["yte"]).mean())


def worker_eval(job):
    seed, members = job
    return members, _accuracy_for(seed, members)


class RetrainGame:
    """Ίδια διεπαφή με το RoundGame (n, value, values, empty_value, grand_value, n_evals)."""

    def __init__(self, seed: int, player_ids, pool=None):
        self.seed = int(seed)
        self.ids = [int(i) for i in player_ids]
        self.n = len(self.ids)
        self.pool = pool
        self._cache: Dict[Tuple[int, ...], float] = {}
        self.n_evals = 0

    def _members(self, mask) -> Tuple[int, ...]:
        return tuple(self.ids[i] for i in np.flatnonzero(np.asarray(mask, dtype=bool)))

    def values(self, masks) -> np.ndarray:
        # Ο KernelSHAP ζητά όλους τους συνασπισμούς με μία κλήση, οπότε οι νέοι
        # εκπαιδεύονται παράλληλα.
        masks = np.atleast_2d(np.asarray(masks))
        keys = [self._members(m) for m in masks]
        jobs = [(self.seed, k) for k in sorted({k for k in keys if k not in self._cache})]
        results = (self.pool.imap_unordered(worker_eval, jobs, chunksize=1)
                   if self.pool is not None else map(worker_eval, jobs))
        for k, v in results:
            self._cache[k] = v
            self.n_evals += 1
        return np.array([self._cache[k] for k in keys], dtype=np.float64)

    def value(self, mask) -> float:
        return float(self.values(np.asarray(mask)[None, :])[0])

    def empty_value(self) -> float:
        return self.value(np.zeros(self.n, dtype=bool))

    def grand_value(self) -> float:
        return self.value(np.ones(self.n, dtype=bool))

    def span(self) -> float:
        return self.grand_value() - self.empty_value()
