# Επιλογή εκτιμητή Shapley από το όνομά του, με τις παραμέτρους του.

from typing import Optional

import numpy as np

from implementation.contribution.sv.estimators import ESTIMATORS


class ShapleyEstimator:
    def __init__(self, estimator_alg: str = "kernelshap", params: Optional[dict] = None):
        if estimator_alg not in ESTIMATORS:
            raise NotImplementedError(f"άγνωστος εκτιμητής {estimator_alg!r}, διαθέσιμοι {sorted(ESTIMATORS)}")
        self.alg = estimator_alg
        self.params = {} if params is None else dict(params)
        self._impl = ESTIMATORS[estimator_alg](**self.params)

    def __repr__(self):
        return repr(self._impl)

    def estimate(self, game, rng: Optional[np.random.Generator] = None) -> dict:
        out = self._impl.estimate(game, rng)
        out["estimator"] = self.alg
        return out
