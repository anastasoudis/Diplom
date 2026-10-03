# Από τις τιμές φ σε μερίδια ανταμοιβής που αθροίζουν στο 1.
#
# Με αρνητικά φ ο λόγος φ_i / Σφ δίνει αρνητικά μερίδια, οπότε χρειάζεται πολιτική.
# Προτάσεις είναι η εκθετική (Jaynes 1957) και η αφινική (Tastan κ.ά. 2024, Εξ. 6).
# Οι raw, zero_clip και rank είναι βάσεις σύγκρισης. Το αρχείο δεν
# σχεδιάζει μηχανισμό κινήτρων. Μετατρέπει μόνο ένα διάνυσμα φ σε μερίδια.

from typing import Optional

import numpy as np


def _normalize(w: np.ndarray) -> np.ndarray:
    """Μερίδια που αθροίζουν στο 1, ή ίσα μερίδια αν το άθροισμα είναι μηδέν."""
    w = np.asarray(w, dtype=np.float64)
    t = w.sum()
    if not np.isfinite(t) or abs(t) < 1e-15:
        return np.full(len(w), 1.0 / len(w))
    return w / t


class RawShare:
    """φ_i / Σφ. Με αρνητικά φ δίνει αρνητικά μερίδια. Με Σφ κοντά στο 0 γίνεται ασταθής."""

    def __repr__(self):
        return "RawShare()"

    def allocate(self, phi: np.ndarray) -> dict:
        phi = np.asarray(phi, dtype=np.float64)
        w = _normalize(phi)
        warn = []
        if (w < 0).any():
            warn.append(f"{int((w < 0).sum())} αρνητικά μερίδια")
        if abs(phi.sum()) < 0.1 * np.abs(phi).sum():
            warn.append("το Σφ είναι μικρό σε σχέση με τα |φ|, ασταθές")
        return {"share": w, "excluded": [], "warnings": warn}


class ZeroClip:
    """max(φ, 0) και κανονικοποίηση. Όλοι οι αρνητικοί παίρνουν μηδέν και δεν διακρίνονται μεταξύ τους."""

    def __repr__(self):
        return "ZeroClip()"

    def allocate(self, phi: np.ndarray) -> dict:
        phi = np.asarray(phi, dtype=np.float64)
        return {"share": _normalize(np.maximum(phi, 0.0)),
                "excluded": [int(i) for i in np.flatnonzero(phi < 0)],
                "warnings": ["οι αρνητικοί δεν διακρίνονται μεταξύ τους"]}


class AffineShift:
    """γ_i = (1 + z_i) / 2 και κανονικοποίηση, από την Εξ. (6) των Tastan κ.ά. (2024).

    Εκεί η τιμή είναι συνημίτονο, άρα βρίσκεται ήδη στο [-1, 1]. Το φ εδώ δεν είναι
    φραγμένο, οπότε πρώτα κλιμακώνεται με έναν από δύο τρόπους:
      maxabs  z = φ / max|φ|, οπότε το φ = 0 αντιστοιχεί σε γ = 0,5
      minmax  z = 2 (φ - min) / (max - min) - 1, οπότε ο χειρότερος παίρνει γ = 0
    Και οι δύο είναι αφινικοί μετασχηματισμοί του φ, οπότε μετά την κανονικοποίηση
    διαφέρουν μόνο στο πού πέφτει το μηδέν (-max|φ| και min φ αντίστοιχα). Δίνουν τα
    ίδια μερίδια όταν ο client με το μεγαλύτερο |φ| έχει αρνητικό φ.
    """

    def __init__(self, scale: str = "maxabs"):
        if scale not in ("maxabs", "minmax"):
            raise NotImplementedError(f"άγνωστη κλιμάκωση {scale!r}")
        self.scale = scale

    def __repr__(self):
        return f"AffineShift(scale={self.scale!r})"

    def allocate(self, phi: np.ndarray) -> dict:
        phi = np.asarray(phi, dtype=np.float64)
        if self.scale == "maxabs":
            m = np.max(np.abs(phi))
            z = phi / m if m > 0 else np.zeros_like(phi)
        else:
            lo, hi = phi.min(), phi.max()
            z = (2 * (phi - lo) / (hi - lo) - 1) if hi > lo else np.zeros_like(phi)
        share = _normalize((1.0 + z) / 2.0)
        n_zero = int((share == 0).sum())
        return {"share": share, "excluded": [],
                "warnings": [f"{n_zero} από τους {len(share)} παίρνουν ακριβώς μηδέν" if n_zero
                             else "κανείς δεν παίρνει μηδέν"]}


class RankShare:
    """Μερίδιο ανάλογο της θέσης στην κατάταξη και όχι της τιμής.

    Η κατάταξη των clients μένει πιο σταθερή από τις τιμές τους, αλλά χάνεται το
    πόσο απέχει ο ένας από τον άλλον.
    """

    def __repr__(self):
        return "RankShare()"

    def allocate(self, phi: np.ndarray) -> dict:
        order = np.argsort(np.argsort(np.asarray(phi, dtype=np.float64)))   # 0 ο χειρότερος
        return {"share": _normalize(order + 1.0), "excluded": [],
                "warnings": ["αγνοεί το μέγεθος των διαφορών"]}


class ExponentialShare:
    """γ_i ανάλογο του exp(β z_i), δηλαδή softmax πάνω στην κλιμακωμένη συνεισφορά.

    Η εκθετική μορφή είναι η κατανομή μέγιστης εντροπίας υπό περιορισμό στη μέση
    τιμή (Jaynes 1957, σελ. 623, Εξ. 2-10). Δίνει θετικό βάρος σε κάθε client, ακόμη
    και στους αρνητικούς. Το β ρυθμίζει πόση ανισότητα επιτρέπεται. Με β κοντά στο 0
    όλοι παίρνουν ίσα, ενώ με μεγάλο β τα παίρνει σχεδόν όλα ο πρώτος.

    Κλιμάκωση πριν από το exp:
      maxabs  z = φ / max|φ| στο [-1, 1], οπότε ο λόγος μεριδίων των δύο άκρων είναι e^(2β)
      std     z = φ / τ.α.(φ)
      none    z = φ. Μόνο αυτή η εκδοχή μένει ίδια αν προστεθεί μια σταθερά σε όλα τα φ.
    """

    def __init__(self, beta: float = 1.0, scale: str = "maxabs"):
        if scale not in ("maxabs", "std", "none"):
            raise NotImplementedError(f"άγνωστη κλιμάκωση {scale!r}")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError(f"β = {beta}, πρέπει να είναι πεπερασμένο και μη αρνητικό")
        self.beta = float(beta)
        self.scale = scale

    def __repr__(self):
        return f"ExponentialShare(beta={self.beta!r}, scale={self.scale!r})"

    def allocate(self, phi: np.ndarray) -> dict:
        phi = np.asarray(phi, dtype=np.float64)
        if self.scale == "maxabs":
            d = np.max(np.abs(phi))
        elif self.scale == "std":
            d = float(np.std(phi, ddof=1))
        else:
            d = 1.0
        z = phi / d if d > 0 else np.zeros_like(phi)
        # η αφαίρεση του μεγίστου αποφεύγει υπερχείλιση και δεν αλλάζει τα μερίδια
        share = _normalize(np.exp(self.beta * (z - z.max())))
        if (share <= 0).any():
            warn = ["το β είναι τόσο μεγάλο που κάποια μερίδια στρογγυλοποιήθηκαν σε μηδέν"]
        else:
            warn = [f"κανείς δεν παίρνει μηδέν (λόγος άκρων {share.max() / share.min():.1f})"]
        return {"share": share, "excluded": [], "warnings": warn}


POLICIES = {"raw": RawShare, "zero_clip": ZeroClip, "affine": AffineShift,
            "rank": RankShare, "exponential": ExponentialShare}


class RewardPolicy:
    def __init__(self, policy: str = "affine", params: Optional[dict] = None):
        if policy not in POLICIES:
            raise NotImplementedError(f"άγνωστη πολιτική {policy!r}, διαθέσιμες {sorted(POLICIES)}")
        self.alg = policy
        self.params = {} if params is None else dict(params)
        self._impl = POLICIES[policy](**self.params)

    def __repr__(self):
        return repr(self._impl)

    def allocate(self, phi) -> dict:
        out = self._impl.allocate(phi)
        out["policy"] = self.alg
        s = np.asarray(out["share"], dtype=np.float64)
        if not np.isclose(s.sum(), 1.0, atol=1e-9):
            raise AssertionError(f"η πολιτική {self.alg!r} έδωσε μερίδια με άθροισμα {s.sum():.6f}")
        return out
