# Η συνάθροιση FedAvg (McMahan κ.ά. 2017).

from functools import reduce
from typing import List, Tuple

import numpy as np


def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """Μέσος όρος των παραμέτρων των clients, σταθμισμένος με το πλήθος των samples τους."""
    total = sum(n for _, n in results)
    weighted = [[layer * n for layer in w] for w, n in results]
    return [reduce(np.add, layers) / total for layers in zip(*weighted)]
