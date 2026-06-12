# =============================================================================
# ch04_fl_implementation / incentive.py
# -----------------------------------------------------------------------------
# Μηχανισμός κινήτρων βασισμένος σε Stackelberg game και LOO contribution.
#
# Ο server (leader) ορίζει το reward budget.
# Οι clients (followers) επιλέγουν προσπάθεια e_i που μεγιστοποιεί
# την ατομική τους ωφέλεια: u_i(e_i) = r_i · q_i(e_i) - c_i(e_i)
#
# Συνεισφορά client i μετράται με LOO (Leave-One-Out):
#   φ̂_i = v(N) - v(N \ {i})
# όπου v(S) = accuracy του global model εκπαιδευμένου με clients S.
#
# Αντιστοιχεί σε: §4.3 (Incentive Mechanism) της διπλωματικής
# =============================================================================

import numpy as np
from typing import List, Dict, Tuple

from implementation.ch04_fl_implementation.model import get_model
from implementation.ch04_fl_implementation.client import FLClient
from implementation.ch04_fl_implementation.server import FLServer


# --- Παράμετροι μηχανισμού --------------------------------------------------

TOTAL_BUDGET    = 1.0     # συνολικό budget R που διαθέτει ο server
COST_COEFF      = 0.5     # συντελεστής κόστους c — c_i(e_i) = c · e_i²
LOO_ROUNDS      = 3       # γύροι FL για κάθε LOO αξιολόγηση (trade-off: ταχύτητα vs ακρίβεια)
MIN_REWARD      = 0.01    # ελάχιστη αμοιβή ανεξαρτήτως συνεισφοράς


# --- Υπολογισμός συνεισφοράς ------------------------------------------------

def compute_loo_contributions(
    clients: List[FLClient],
    n_rounds: int = LOO_ROUNDS,
    device: str = "cpu",
) -> Dict[int, float]:
    """Υπολογίζει LOO contribution για κάθε client.

    LOO (Leave-One-Out): η αξία του client i είναι πόσο χάνεται
    αν τον αφαιρέσουμε από το σύνολο N:
        φ̂_i = v(N) - v(N \ {i})

    όπου v(S) = accuracy global model μετά από n_rounds FL με clients S.

    Είναι πολυπλοκότητας O(|N| · n_rounds · LOCAL_EPOCHS) — αποδεκτό
    για αριθμό clients < 50. Για μεγαλύτερα σύνολα → Shapley sampling.

    Returns:
        dict client_id → LOO contribution (≥ 0)
    """
    print(f"Computing LOO contributions ({len(clients)} clients, "
          f"{n_rounds} rounds each)...")

    # βήμα 1: v(N) — accuracy όλων των clients μαζί
    v_all = _train_and_eval(clients, n_rounds, device)
    print(f"  v(N) = {v_all:.4f}  [all {len(clients)} clients]")

    # βήμα 2: v(N \ {i}) για κάθε client i
    contributions: Dict[int, float] = {}
    for i, client in enumerate(clients):
        subset = [c for c in clients if c.id != client.id]
        v_without_i = _train_and_eval(subset, n_rounds, device)
        phi_i = max(0.0, v_all - v_without_i)   # clip at 0 — αρνητική "συνεισφορά" = 0
        contributions[client.id] = phi_i
        print(f"  v(N\\{{{client.id}}}) = {v_without_i:.4f}  "
              f"→ φ̂_{client.id} = {phi_i:.4f}")

    return contributions


def _train_and_eval(clients: List[FLClient],
                    n_rounds: int,
                    device: str) -> float:
    """Εκπαιδεύει FL με τους δοσμένους clients και επιστρέφει global accuracy.

    Εσωτερική βοηθητική — χρησιμοποιείται μόνο από compute_loo_contributions.
    """
    if not clients:
        return 0.0

    model  = get_model(device)
    server = FLServer(clients, device)
    server.model = model

    for r in range(1, n_rounds + 1):
        server.run_round(r, fraction=1.0)   # χρησιμοποίησε όλους τους clients

    _, accuracy = server.evaluate_global()
    return accuracy


# --- Κατανομή αμοιβών -------------------------------------------------------

def allocate_rewards(
    contributions: Dict[int, float],
    total_budget: float = TOTAL_BUDGET,
    min_reward: float = MIN_REWARD,
) -> Dict[int, float]:
    """Κατανέμει το budget R ανάλογα με τη συνεισφορά κάθε client.

    Κανόνας κατανομής (αναλογική προς LOO):
        r_i = max(min_reward, φ̂_i / Σ_j φ̂_j · R)

    Clients με μηδενική συνεισφορά λαμβάνουν μόνο min_reward.
    Το budget αναδιανέμεται ώστε το άθροισμα να παραμένει ≤ R.

    Returns:
        dict client_id → reward ∈ [min_reward, R]
    """
    total_contrib = sum(contributions.values())
    rewards: Dict[int, float] = {}

    if total_contrib == 0:
        # κανείς δεν συνεισέφερε — ισόποση κατανομή min_reward
        for cid in contributions:
            rewards[cid] = min_reward
        return rewards

    for cid, phi in contributions.items():
        prop_reward = (phi / total_contrib) * total_budget
        rewards[cid] = max(min_reward, prop_reward)

    return rewards


# --- Βέλτιστη προσπάθεια clients (Stackelberg) -------------------------------

def optimal_effort(reward: float, cost_coeff: float = COST_COEFF) -> float:
    """Βέλτιστη προσπάθεια client δεδομένης αμοιβής r και κόστους c·e².

    Το πρόβλημα μεγιστοποίησης ωφέλειας:
        max_e  u(e) = r · e - c · e²

    Παράγουγος: du/de = r - 2c·e = 0  →  e* = r / (2c)

    Args:
        reward:     αμοιβή r_i που έχει ορίσει ο server για τον client
        cost_coeff: παράμετρος κόστους c (ίδια για όλους — homogeneous clients)

    Returns:
        Βέλτιστη προσπάθεια e* ∈ [0, 1] (κλιπαρισμένο στο [0,1])
    """
    e_star = reward / (2 * cost_coeff)
    return float(np.clip(e_star, 0.0, 1.0))   # η προσπάθεια είναι ποσοστό [0,1]


def stackelberg_equilibrium(
    contributions: Dict[int, float],
    total_budget: float = TOTAL_BUDGET,
    cost_coeff: float   = COST_COEFF,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Υπολογίζει Stackelberg ισορροπία (rewards, efforts).

    Ο server (leader) ορίζει rewards αναλογικά προς τη συνεισφορά.
    Κάθε client (follower) απαντά με βέλτιστη προσπάθεια e* = r/(2c).

    Returns:
        (rewards, efforts): dicts client_id → float
    """
    rewards = allocate_rewards(contributions, total_budget)
    efforts = {cid: optimal_effort(r, cost_coeff)
               for cid, r in rewards.items()}
    return rewards, efforts


# --- Κεντρική συνάρτηση -----------------------------------------------------

def run_incentive_mechanism(
    clients: List[FLClient],
    n_loo_rounds: int = LOO_ROUNDS,
    total_budget: float = TOTAL_BUDGET,
    device: str = "cpu",
) -> Dict:
    """Εκτελεί ολόκληρο τον κύκλο του incentive mechanism.

    Βήματα:
        1. Υπολογισμός LOO contributions
        2. Κατανομή rewards (αναλογικά)
        3. Υπολογισμός Stackelberg ισορροπίας (optimal efforts)
        4. Εκτύπωση αποτελεσμάτων

    Returns:
        dict με contributions, rewards, efforts
    """
    print("\n" + "=" * 55)
    print("INCENTIVE MECHANISM — Stackelberg + LOO")
    print("=" * 55)

    contributions = compute_loo_contributions(clients, n_loo_rounds, device)
    rewards, efforts = stackelberg_equilibrium(contributions, total_budget)

    print("\nResults:")
    print(f"{'Client':>8}  {'φ̂_i':>8}  {'r_i':>8}  {'e*_i':>8}")
    print("-" * 40)
    for cid in sorted(contributions.keys()):
        print(f"{cid:>8}  {contributions[cid]:>8.4f}  "
              f"{rewards[cid]:>8.4f}  {efforts[cid]:>8.4f}")

    total_rewards = sum(rewards.values())
    print(f"\nTotal rewards paid: {total_rewards:.4f} / {total_budget:.4f}")

    return {
        "contributions": contributions,
        "rewards":       rewards,
        "efforts":       efforts,
    }
