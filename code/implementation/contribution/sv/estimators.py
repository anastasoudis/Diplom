# Οι εκτιμητές Shapley. Όλοι έχουν τη μέθοδο estimate(game, rng) και εκτιμούν το ίδιο
# μέγεθος, την τιμή Shapley κάθε παίκτη στο παίγνιο του coalition.py. Διαφέρουν μόνο
# στον τρόπο που την προσεγγίζουν, γι' αυτό η συμφωνία τους ελέγχει την εκτίμηση.
#
#   KernelShapEstimator        KernelSHAP της βιβλιοθήκης shap (Lundberg και Lee 2017)
#   GtgShapleyEstimator        GTG-Shapley χωρίς guided sampling, το GTG-Tib (Liu κ.ά. 2022)
#   GtgGuidedShapleyEstimator  GTG-Shapley με guided sampling
#   ExactEstimator             ο ορισμός, με όλους τους 2^n συνασπισμούς
#   *ClasswiseEstimator        οι ίδιοι με διανυσματική αξία

import math
from typing import Optional

import numpy as np

KSHAP_L1_MODES = ("off", "num_features")


def kshap_l1_reg(mode: str, n: int):
    """Η τιμή του l1_reg του shap.

    off           καμία επιλογή παικτών. Όλοι παίρνουν τιμή από την παλινδρόμηση του
                  Θεωρήματος 2 (Lundberg και Lee 2017), δηλαδή ο ορισμός.
    num_features  η βιβλιοθήκη κρατά έως n παίκτες. Σε παίκτη με σχεδόν μηδενική
                  συνεισφορά μπορεί να δώσει 0 ακριβώς.
    """
    if mode == "off":
        return False
    if mode == "num_features":
        return f"num_features({n})"
    raise ValueError(f"άγνωστη ρύθμιση {mode!r}, διαθέσιμες {KSHAP_L1_MODES}")


class KernelShapEstimator:
    """Ο shap.KernelExplainer με παίκτες τους clients αντί για χαρακτηριστικά.

    Το background είναι το διάνυσμα μηδενικών (κανείς δεν συμμετέχει, v(∅)) και το
    instance το διάνυσμα μονάδων (όλοι συμμετέχουν, v(N)). Ο explainer ζητά την αξία
    συνασπισμών και λύνει τη σταθμισμένη παλινδρόμηση του Θεωρήματος 2.
    """

    def __init__(self, nsamples="auto", l1_reg=False, seed: int = 0):
        self.nsamples = nsamples
        self.l1_reg = l1_reg
        self.seed = seed
        self._calls = 0

    def __repr__(self):
        return f"KernelShapEstimator(nsamples={self.nsamples!r}, l1_reg={self.l1_reg!r})"

    def _explain(self, game):
        import shap
        n = game.n
        # Ο shap διαλέγει συνασπισμούς με την καθολική γεννήτρια του numpy και δεν
        # δέχεται seed. Η γεννήτρια σπέρνεται εδώ και επανέρχεται μετά, ώστε το
        # αποτέλεσμα να αναπαράγεται και να μην επηρεάζεται τίποτα άλλο. Κάθε κλήση
        # παίρνει άλλο seed, ώστε κάθε γύρος να δειγματοληπτεί άλλους συνασπισμούς και
        # τα σφάλματα των γύρων να μην είναι συσχετισμένα στο άθροισμα.
        state = np.random.get_state()
        np.random.seed((self.seed * 1_000_003 + self._calls) % (2 ** 31 - 1))
        self._calls += 1
        explainer = shap.KernelExplainer(game.values, np.zeros((1, n)))
        raw = explainer.shap_values(np.ones((1, n)), nsamples=self.nsamples,
                                    l1_reg=self.l1_reg, silent=True)
        np.random.set_state(state)
        return raw

    def estimate(self, game, rng: Optional[np.random.Generator] = None) -> dict:
        phi = np.asarray(self._explain(game), dtype=np.float64).reshape(-1)[:game.n]
        v0, vN = game.empty_value(), game.grand_value()
        return {"phi": phi, "v_empty": v0, "v_grand": vN,
                "efficiency_gap": float(vN - v0 - phi.sum()),
                "n_evals": game.n_evals, "iters": None, "converged": None}


class GtgShapleyEstimator:
    """GTG-Shapley (Liu κ.ά. 2022, Algorithm 1) χωρίς guided sampling.

    1. Αν |v(N) - v(∅)| <= eps_between, ο γύρος παραλείπεται (between-round truncation).
    2. Δειγματοληψία τυχαίων μεταθέσεων των παικτών.
    3. Μέσα σε μια μετάθεση, μόλις η αξία πλησιάσει το v(N) κατά λιγότερο από
       eps_within, οι υπόλοιποι παίρνουν οριακή συνεισφορά 0 χωρίς αξιολόγηση.
    4. Διακοπή όταν η σχετική μεταβολή των φ στις τελευταίες conv_window μεταθέσεις
       πέσει κάτω από conv_tol, ή μετά από max_iters μεταθέσεις.
    Το άρθρο ονομάζει αυτή την εκδοχή GTG-Tib.
    """

    def __init__(self, eps_between: float = 1e-4, eps_within: float = 1e-4,
                 max_iters: int = 100, min_iters: int = 10,
                 conv_tol: float = 0.05, conv_window: int = 10, truncate: bool = True):
        self.eps_between = eps_between
        self.eps_within = eps_within
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.conv_tol = conv_tol
        self.conv_window = conv_window
        self.truncate = truncate

    def __repr__(self):
        return (f"GtgShapleyEstimator(eps_b={self.eps_between}, eps_i={self.eps_within}, "
                f"max_iters={self.max_iters})")

    def _permutation(self, rng, n: int, k: int) -> np.ndarray:
        """Η σειρά των παικτών στη μετάθεση k. Μία κλήση της rng ανά μετάθεση."""
        return rng.permutation(n)

    def estimate(self, game, rng: Optional[np.random.Generator] = None) -> dict:
        rng = rng or np.random.default_rng(0)
        n = game.n
        v0, vN = game.empty_value(), game.grand_value()
        phi = np.zeros(n, dtype=np.float64)
        if self.truncate and abs(vN - v0) <= self.eps_between:
            return {"phi": phi, "v_empty": v0, "v_grand": vN,
                    "efficiency_gap": float(vN - v0), "n_evals": game.n_evals,
                    "iters": 0, "converged": True, "skipped": True}

        total = np.zeros(n, dtype=np.float64)
        history, k, converged = [], 0, False
        while k < self.max_iters:
            k += 1
            perm = self._permutation(rng, n, k - 1)
            mask = np.zeros(n, dtype=bool)
            v_prev = v0
            for player in perm:
                mask[player] = True
                if self.truncate and abs(vN - v_prev) < self.eps_within:
                    v_cur = v_prev
                else:
                    v_cur = game.value(mask)
                total[player] += v_cur - v_prev
                v_prev = v_cur
            phi = total / k
            history.append(phi.copy())
            if k >= max(self.min_iters, self.conv_window + 1):
                w = history[-self.conv_window:]
                den = np.maximum(np.abs(phi), 1e-12)
                rel = np.mean([np.mean(np.abs(h - phi) / den) for h in w[:-1]])
                if rel < self.conv_tol:
                    converged = True
                    break
        return {"phi": phi, "v_empty": v0, "v_grand": vN,
                "efficiency_gap": float(vN - v0 - phi.sum()),
                "n_evals": game.n_evals, "iters": k, "converged": converged, "skipped": False}


class GtgGuidedShapleyEstimator(GtgShapleyEstimator):
    """GTG-Shapley με guided sampling (Liu κ.ά. 2022, γραμμή 7 του Algorithm 1).

    Οι πρώτες m θέσεις κάθε μετάθεσης δεν είναι τυχαίες. Περνούν με σταθερή σειρά από
    όλες τις διατάξεις m παικτών, ώστε κάθε παίκτης να βρεθεί στις πρώτες θέσεις
    περίπου ίσες φορές. Οι πρώτες θέσεις είναι αυτές που αξιολογούνται πραγματικά,
    γιατί το within-round truncation μηδενίζει τις τελευταίες. Οι υπόλοιπες n - m θέσεις
    είναι τυχαία μετάθεση των υπολοίπων.

    Το άρθρο δεν δίνει τιμή για το m ούτε τη σειρά απαρίθμησης. Εδώ m = 2 και η
    μετάθεση k έχει στην πρώτη θέση τον παίκτη k mod n. Οι επόμενες θέσεις απέχουν από
    αυτόν κατά διαφορετικές μετατοπίσεις από 1 έως n-1, που αλλάζουν σε κάθε
    βήμα. Έτσι καλύπτονται όλες οι n(n-1) διατάξεις πριν επαναληφθεί κάποια. Με m <= 3
    και n = 21 η κάλυψη είναι πλήρης, γιατί τα 21, 20, 19 είναι ανά δύο πρώτα μεταξύ τους.

    Σε κάθε γύρο η απαρίθμηση ξεκινά από τυχαίο σημείο (decorrelate). Χωρίς αυτό, κάθε
    γύρος θα δειγματοληπτούσε τις ίδιες διατάξεις και τα σφάλματα των γύρων δεν θα
    αλληλοαναιρούνταν στο άθροισμα.
    """

    def __init__(self, eps_between: float = 1e-4, eps_within: float = 1e-4,
                 max_iters: int = 100, min_iters: int = 10, conv_tol: float = 0.05,
                 conv_window: int = 10, truncate: bool = True, m: int = 2, decorrelate: bool = True):
        super().__init__(eps_between, eps_within, max_iters, min_iters, conv_tol, conv_window, truncate)
        if not 1 <= int(m) <= 3:
            raise ValueError(f"m = {m}: η απαρίθμηση εδώ καλύπτει όλες τις διατάξεις μόνο για 1 <= m <= 3")
        self.m = int(m)
        self.decorrelate = bool(decorrelate)
        self._offset = 0

    def __repr__(self):
        return (f"GtgGuidedShapleyEstimator(eps_b={self.eps_between}, eps_i={self.eps_within}, "
                f"max_iters={self.max_iters}, m={self.m}, decorrelate={self.decorrelate})")

    def estimate(self, game, rng=None):
        rng = rng or np.random.default_rng(0)
        if self.decorrelate:
            period = 1
            for j in range(min(self.m, game.n)):
                period *= (game.n - j)
            self._offset = int(rng.integers(0, period))
        else:
            self._offset = 0
        return super().estimate(game, rng)

    @staticmethod
    def _prefix(n: int, m: int, k: int) -> list:
        """Οι πρώτες m θέσεις της μετάθεσης k."""
        i0 = k % n
        pool = list(range(1, n))
        offsets = []
        for _ in range(m - 1):
            offsets.append(pool.pop(k % len(pool)))
        return [i0] + [(i0 + d) % n for d in offsets]

    def _permutation(self, rng, n: int, k: int) -> np.ndarray:
        head = self._prefix(n, min(self.m, n), k + self._offset)
        rest = np.array([p for p in range(n) if p not in set(head)], dtype=int)
        rng.shuffle(rest)
        return np.concatenate([np.array(head, dtype=int), rest])


class ExactEstimator:
    """Η τιμή Shapley από τον ορισμό, με όλους τους 2^n συνασπισμούς (Shapley 1953).

    φ_i = Σ_{S ∌ i} |S|! (n - |S| - 1)! / n! [v(S ∪ {i}) - v(S)]

    Κάθε συνασπισμός S αξιολογείται μία φορά και συνεισφέρει θετικά στα φ των μελών
    του και αρνητικά στα φ των υπολοίπων.
    """

    def __init__(self, max_players: int = 21):
        self.max_players = max_players

    def __repr__(self):
        return f"ExactEstimator(max_players={self.max_players})"

    def estimate(self, game, rng: Optional[np.random.Generator] = None) -> dict:
        n = game.n
        if n > self.max_players:
            raise ValueError(f"{n} παίκτες σημαίνει {2 ** n:,} συνασπισμούς, όριο {self.max_players}")
        phi = np.zeros(n, dtype=np.float64)
        fact = math.factorial
        for bits in range(1 << n):
            mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool)
            s = int(mask.sum())
            v = game.value(mask)
            for i in range(n):
                if mask[i]:
                    phi[i] += fact(s - 1) * fact(n - s) / fact(n) * v
                else:
                    phi[i] -= fact(s) * fact(n - s - 1) / fact(n) * v
        v0, vN = game.empty_value(), game.grand_value()
        return {"phi": phi, "v_empty": v0, "v_grand": vN,
                "efficiency_gap": float(vN - v0 - phi.sum()),
                "n_evals": game.n_evals, "iters": None, "converged": True}


class KernelShapClasswiseEstimator(KernelShapEstimator):
    """KernelSHAP με διανυσματική αξία, μία στήλη φ ανά δραστηριότητα ή άτομο.

    Ο KernelExplainer λύνει μία παλινδρόμηση ανά έξοδο πάνω στους ίδιους
    συνασπισμούς, οπότε οι αξιολογήσεις γίνονται μία φορά για όλες τις στήλες.
    """

    def __repr__(self):
        return f"KernelShapClasswiseEstimator(nsamples={self.nsamples!r}, l1_reg={self.l1_reg!r})"

    def estimate(self, game, rng: Optional[np.random.Generator] = None) -> dict:
        n, M = game.n, game.n_outputs
        raw = self._explain(game)
        # Ανάλογα με την έκδοση, ο shap επιστρέφει λίστα M πινάκων (1, n) ή πίνακα
        # (1, n, M). Και τα δύο φέρνονται σε (n, M).
        if isinstance(raw, list):
            phi = np.stack([np.asarray(a, dtype=np.float64).reshape(-1)[:n] for a in raw], axis=1)
        else:
            a = np.asarray(raw, dtype=np.float64)
            if not (a.shape[-1] == M and a.size == n * M):
                raise ValueError(f"απρόσμενο σχήμα από τον shap {a.shape}, n={n}, M={M}")
            phi = a.reshape(n, M)
        v0, vN = game.empty_value(), game.grand_value()
        return {"phi": phi, "v_empty": v0, "v_grand": vN,
                # η ιδιότητα efficiency ισχύει χωριστά σε κάθε στήλη
                "efficiency_gap": [float(x) for x in (vN - v0 - phi.sum(axis=0))],
                "n_evals": game.n_evals, "iters": None, "converged": None}


class GtgClasswiseEstimator(GtgShapleyEstimator):
    """GTG-Shapley με διανυσματική αξία. Οι οριακές συνεισφορές αθροίζονται ανά στήλη.

    Τα δύο truncation συγκρίνουν αριθμούς με κατώφλι, οπότε με διάνυσμα χρειάζονται
    σύνοψη. Η πύλη του γύρου χρησιμοποιεί το σταθμισμένο άθροισμα, δηλαδή τη μεταβολή
    του accuracy, ώστε να κρατά τους ίδιους γύρους με το βαθμωτό τρέξιμο. Το within-round
    truncation χρησιμοποιεί το μέγιστο πάνω στις στήλες. Με το σταθμισμένο άθροισμα η
    μετάθεση θα σταματούσε όταν σταθεροποιηθεί το συνολικό accuracy, ενώ κάποια
    δραστηριότητα θα απείχε ακόμη από το v(N). Έτσι η efficiency θα χανόταν ανά στήλη.
    """

    def __init__(self, eps_between=1e-4, eps_within=1e-4, max_iters=100, min_iters=10,
                 conv_tol=0.05, conv_window=10, truncate=True):
        super().__init__(eps_between, eps_within, max_iters, min_iters, conv_tol, conv_window, truncate)

    def __repr__(self):
        return (f"GtgClasswiseEstimator(eps_b={self.eps_between}, eps_i={self.eps_within}, "
                f"max_iters={self.max_iters})")

    def estimate(self, game, rng: Optional[np.random.Generator] = None) -> dict:
        rng = rng or np.random.default_rng(0)
        n, M = game.n, game.n_outputs
        w = game.class_weights
        v0, vN = game.empty_value(), game.grand_value()
        phi = np.zeros((n, M), dtype=np.float64)
        if self.truncate and abs(float(w @ (vN - v0))) <= self.eps_between:
            return {"phi": phi, "v_empty": v0, "v_grand": vN,
                    "efficiency_gap": [float(x) for x in (vN - v0)],
                    "n_evals": game.n_evals, "iters": 0, "converged": True, "skipped": True}

        total = np.zeros((n, M), dtype=np.float64)
        history, k, converged = [], 0, False
        while k < self.max_iters:
            k += 1
            perm = self._permutation(rng, n, k - 1)
            mask = np.zeros(n, dtype=bool)
            v_prev = v0
            for player in perm:
                mask[player] = True
                if self.truncate and float(np.max(np.abs(vN - v_prev))) < self.eps_within:
                    v_cur = v_prev
                else:
                    v_cur = game.value(mask)
                total[player] += v_cur - v_prev
                v_prev = v_cur
            phi = total / k
            history.append(phi.copy())
            if k >= max(self.min_iters, self.conv_window + 1):
                win = history[-self.conv_window:]
                den = np.maximum(np.abs(phi), 1e-12)
                rel = np.mean([np.mean(np.abs(h - phi) / den) for h in win[:-1]])
                if rel < self.conv_tol:
                    converged = True
                    break
        return {"phi": phi, "v_empty": v0, "v_grand": vN,
                "efficiency_gap": [float(x) for x in (vN - v0 - phi.sum(axis=0))],
                "n_evals": game.n_evals, "iters": k, "converged": converged, "skipped": False}


ESTIMATORS = {"kernelshap": KernelShapEstimator,
              "gtg": GtgShapleyEstimator,
              "gtg_guided": GtgGuidedShapleyEstimator,
              "exact": ExactEstimator,
              "kernelshap_cw": KernelShapClasswiseEstimator,
              "gtg_cw": GtgClasswiseEstimator}
