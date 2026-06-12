# =============================================================================
# ch03_data_analysis / eda_analysis.py
# -----------------------------------------------------------------------------
# Εξερεύνηση του AAL dataset και παραγωγή των figures της §3.2–§3.3.
# Παράγει 4 PDF στο thesis/figures/ και εκτυπώνει βασικά στατιστικά.
#
# Εκτέλεση (από root του project):
#   python3 implementation/ch03_data_analysis/eda_analysis.py
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # χωρίς GUI — αποθηκεύει απευθείας σε αρχείο
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import groupby
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import jensenshannon


DATA_DIR = "Datasets/aal"
FIG_DIR  = "thesis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Παράμετροι dataset
STRIDE = 64    # βήμα sliding window (50% overlap → 128/2)
WIN    = 128   # δείγματα ανά παράθυρο

ACTIVITY_NAMES = {
    1: "Walking", 2: "Walk Up", 3: "Walk Down",
    4: "Sitting", 5: "Standing", 6: "Laying"
}
# ένα χρώμα ανά κλάση — το ίδιο παλέτα σε όλα τα figures για consistency
COLORS     = ["#2196F3", "#4CAF50", "#8BC34A", "#FF9800", "#F44336", "#9C27B0"]
SHORT_LABS = ["WALK", "W-UP", "W-DN", "SIT", "STAND", "LAY"]

# γενικές ρυθμίσεις matplotlib — serif για να ταιριάζει με LaTeX
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":     150,
    "savefig.bbox":   "tight",
    "savefig.pad_inches": 0.05,
})


# =============================================================================
# Φόρτωση δεδομένων
# =============================================================================

print("Loading data...")
X_train   = np.loadtxt(f"{DATA_DIR}/final_X_train.txt", delimiter=",")
y_train   = np.loadtxt(f"{DATA_DIR}/final_y_train.txt", delimiter=",", dtype=int)
X_test    = np.loadtxt(f"{DATA_DIR}/final_X_test.txt",  delimiter=",")
y_test    = np.loadtxt(f"{DATA_DIR}/final_y_test.txt",  delimiter=",", dtype=int)
acc_train = np.loadtxt(f"{DATA_DIR}/final_acc_train.txt", delimiter=",")  # raw accel, shape (T, 3)

print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"  Raw acc signal: {acc_train.shape[0]} rows (continuous, not windowed)")


# =============================================================================
# Εξαγωγή subject IDs
# =============================================================================
# Το dataset δεν έχει στήλη "subject". Αντ' αυτού, τα δεδομένα είναι
# αποθηκευμένα σε blocks: subject_0 × [6 activities], subject_1 × [...], ...
# Βρίσκουμε τα "runs" (αλλαγές τιμής) και ομαδοποιούμε κάθε 6 σε έναν subject.

runs = [(k, sum(1 for _ in g)) for k, g in groupby(y_train)]
# runs = [(1, 34), (2, 35), (3, 2), (4, 35), ... ]

N_SUBJECTS = len(runs) // 6   # 132 runs / 6 activities = 22 subjects

subject_ids = np.zeros(len(y_train), dtype=int)
pos, subj = 0, 0
for i in range(0, len(runs), 6):
    for _, count in runs[i:i + 6]:
        subject_ids[pos:pos + count] = subj
        pos += count
    subj += 1

print(f"  Training subjects: {N_SUBJECTS} | Avg windows/subject: {len(y_train)/N_SUBJECTS:.1f}")


# =============================================================================
# Figure 1 — Κατανομή κλάσεων
# =============================================================================
print("\n[1/4] Class distribution...")

train_counts = [np.sum(y_train == c) for c in range(1, 7)]
test_counts  = [np.sum(y_test  == c) for c in range(1, 7)]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(6)
w = 0.38

bars_tr = ax.bar(x - w/2, train_counts, w, label="Train (n=4252)",
                 color=COLORS, alpha=0.85, edgecolor="white", linewidth=0.5)
bars_te = ax.bar(x + w/2, test_counts,  w, label="Test  (n=1492)",
                 color=COLORS, alpha=0.45, edgecolor="white", linewidth=0.5, hatch="//")

# εκτύπωση αριθμού πάνω από κάθε μπάρα
for bar in list(bars_tr) + list(bars_te):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([ACTIVITY_NAMES[c] for c in range(1, 7)], rotation=15, ha="right")
ax.set_ylabel("Αριθμός παραθύρων")
ax.set_xlabel("Κλάση δραστηριότητας")
ax.set_title("Κατανομή κλάσεων — AAL Dataset")
ax.legend(framealpha=0.9)
ax.set_ylim(0, max(train_counts) * 1.15)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/eda_class_distribution.pdf")
plt.close()
print("  → eda_class_distribution.pdf")


# =============================================================================
# Figure 2 — Χρονοσειρές σήματος ανά κλάση
# =============================================================================
print("\n[2/4] Signal visualization...")

# Για κάθε κλάση βρίσκουμε ένα "καλό" παράθυρο — παρακάμπτουμε τα πρώτα 10
# γιατί μπορεί να αντιστοιχούν στη μεταβατική στιγμή εκκίνησης.
first_window = {c: np.where(y_train == c)[0][10] for c in range(1, 7)}

t = np.arange(WIN) / 50.0   # άξονας χρόνου σε δευτερόλεπτα

fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
axes = axes.flatten()

for i, c in enumerate(range(1, 7)):
    ax = axes[i]
    start = first_window[c] * STRIDE
    end   = start + WIN

    # fallback αν το παράθυρο ξεπερνά το μήκος του signal
    seg = acc_train[start:end] if end <= len(acc_train) else acc_train[:WIN]

    ax.plot(t, seg[:, 0], color="#2196F3", lw=1.0, label="X", alpha=0.9)
    ax.plot(t, seg[:, 1], color="#F44336", lw=1.0, label="Y", alpha=0.9)
    ax.plot(t, seg[:, 2], color="#4CAF50", lw=1.0, label="Z", alpha=0.9)

    ax.set_title(ACTIVITY_NAMES[c], fontweight="bold", pad=4)
    ax.set_ylim(-15, 15)
    ax.axhline(0, color="gray", lw=0.4, linestyle="--")
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if i >= 4:
        ax.set_xlabel("Χρόνος (s)")
    if i % 2 == 0:
        ax.set_ylabel("Επιτάχυνση (m/s²)")

legend_handles = [
    mpatches.Patch(color="#2196F3", label="Άξονας X"),
    mpatches.Patch(color="#F44336", label="Άξονας Y"),
    mpatches.Patch(color="#4CAF50", label="Άξονας Z"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Χρονοσειρές επιταχυνσιόμετρου ανά κλάση δραστηριότητας\n"
    "(παράθυρο 128 δειγμάτων = 2.56 s @ 50 Hz)",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/eda_signals_per_activity.pdf", bbox_inches="tight")
plt.close()
print("  → eda_signals_per_activity.pdf")


# =============================================================================
# Figure 3 — Non-IID heatmap και JSD² ανά subject
# =============================================================================
print("\n[3/4] Non-IID heatmap...")

# κατανομή κλάσεων ανά subject — κάθε γραμμή αθροίζει σε 1
dist_matrix = np.zeros((N_SUBJECTS, 6))
for s in range(N_SUBJECTS):
    counts = np.array([np.sum(y_train[subject_ids == s] == c) for c in range(1, 7)])
    total  = counts.sum()
    dist_matrix[s] = counts / total if total > 0 else counts

# καθολική κατανομή — ο "μέσος" client
global_dist = np.array([np.sum(y_train == c) for c in range(1, 7)], dtype=float)
global_dist /= global_dist.sum()

# JSD² ανά subject — η scipy επιστρέφει sqrt(JSD), οπότε υψώνουμε στο τετράγωνο
jsd = np.array([jensenshannon(dist_matrix[s], global_dist) ** 2
                for s in range(N_SUBJECTS)])

fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(13, 5),
                                       gridspec_kw={"width_ratios": [3, 1]})

# heatmap: κάθε γραμμή = ένας subject, κάθε στήλη = μια κλάση
im = ax_heat.imshow(dist_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.4)
ax_heat.set_xticks(range(6))
ax_heat.set_xticklabels(SHORT_LABS, fontsize=9)
ax_heat.set_yticks(range(N_SUBJECTS))
ax_heat.set_yticklabels([f"S{s+1:02d}" for s in range(N_SUBJECTS)], fontsize=8)
ax_heat.set_xlabel("Κλάση δραστηριότητας")
ax_heat.set_ylabel("Subject (FL Client)")
ax_heat.set_title("Κατανομή κλάσεων ανά subject (Non-IID heatmap)")
plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label="Αναλογία παραθύρων")

# κόκκινο αν JSD² > 0.05 (αξιοσημείωτη ετερογένεια), μπλε αλλιώς
bar_colors = ["#E53935" if j > 0.05 else "#42A5F5" for j in jsd]
ax_bar.barh(range(N_SUBJECTS), jsd, color=bar_colors, alpha=0.85)
ax_bar.axvline(jsd.mean(), color="black", lw=1.2, linestyle="--",
               label=f"Μέσος: {jsd.mean():.3f}")
ax_bar.set_yticks(range(N_SUBJECTS))
ax_bar.set_yticklabels([f"S{s+1:02d}" for s in range(N_SUBJECTS)], fontsize=8)
ax_bar.set_xlabel("JSD²")
ax_bar.set_title("Ετερογένεια\nανά client")
ax_bar.legend(fontsize=9, framealpha=0.9)
ax_bar.grid(axis="x", alpha=0.3, linestyle="--")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/eda_noniid_heatmap.pdf", bbox_inches="tight")
plt.close()
print(f"  → eda_noniid_heatmap.pdf")
print(f"  JSD²: mean={jsd.mean():.4f}, std={jsd.std():.4f}, "
      f"min={jsd.min():.4f}, max={jsd.max():.4f}")


# =============================================================================
# Figure 4 — Αγωγός προ-επεξεργασίας (pipeline diagram)
# =============================================================================
print("\n[4/4] Preprocessing pipeline...")

fig, axes = plt.subplots(1, 5, figsize=(13, 3.5))

# Δημιουργούμε συνθετικό σήμα που μοιάζει με βηματισμό + βαρύτητα
np.random.seed(42)
t_raw  = np.linspace(0, 4, 200)
raw    = (2.5 * np.sin(2 * np.pi * 1.1 * t_raw)    # κίνηση σώματος ~1 Hz
          + 9.8 * np.sin(2 * np.pi * 0.05 * t_raw)  # βαρύτητα ~0.05 Hz
          + 0.3 * np.random.randn(200))              # θόρυβος αισθητήρα

# (1) Raw signal
axes[0].plot(t_raw, raw, color="#1565C0", lw=1.2)
axes[0].set_title("(1) Raw Signal\n(body + gravity)", fontsize=9, fontweight="bold")
axes[0].set_ylabel("m/s²")
axes[0].set_xlabel("t (s)")
axes[0].grid(alpha=0.2)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# (2) Butterworth — χωρισμός gravity / body
nyq = 50 / 2.0
b, a   = butter(3, 0.3 / nyq, btype="low")
gravity = filtfilt(b, a, raw)   # zero-phase: δεν αλλοιώνει χρονική δομή
body    = raw - gravity

axes[1].plot(t_raw, body,    color="#00838F", lw=1.2, label="body")
axes[1].plot(t_raw, gravity, color="#EF6C00", lw=0.9, linestyle="--",
             label="gravity", alpha=0.7)
axes[1].set_title("(2) Butterworth\n(fc=0.3 Hz, n=3)", fontsize=9, fontweight="bold")
axes[1].set_xlabel("t (s)")
axes[1].legend(fontsize=8, framealpha=0.8)
axes[1].grid(alpha=0.2)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# (3) Z-score: κεντράρισμα γύρω από 0, κλίμακα σε μονάδες τυπικής απόκλισης
body_norm = (body - body.mean()) / body.std()
axes[2].plot(t_raw, body_norm, color="#558B2F", lw=1.2)
axes[2].axhline(0,  color="gray", lw=0.4, linestyle="--")
axes[2].axhline(1,  color="gray", lw=0.4, linestyle=":", alpha=0.5)
axes[2].axhline(-1, color="gray", lw=0.4, linestyle=":", alpha=0.5)
axes[2].set_title("(3) Z-score\n(per client)", fontsize=9, fontweight="bold")
axes[2].set_xlabel("t (s)")
axes[2].set_ylabel("σ")
axes[2].grid(alpha=0.2)
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

# (4) Sliding window — χρωματισμός 3 επικαλυπτόμενων παραθύρων
axes[3].plot(t_raw, body_norm, color="#558B2F", lw=1.0, alpha=0.35)
win_colors  = ["#E53935", "#1E88E5", "#43A047"]
win_starts  = [0, 32, 64]   # indices στα 200 δείγματα
for col, ws in zip(win_colors, win_starts):
    we = min(ws + 64, len(t_raw) - 1)
    axes[3].axvspan(t_raw[ws], t_raw[we], alpha=0.12, color=col)
    axes[3].plot(t_raw[ws:ws+64], body_norm[ws:ws+64], color=col, lw=1.5)
axes[3].set_title("(4) Sliding Window\n(W=128, 50% overlap)", fontsize=9, fontweight="bold")
axes[3].set_xlabel("t (s)")
axes[3].grid(alpha=0.2)
axes[3].spines["top"].set_visible(False)
axes[3].spines["right"].set_visible(False)

# (5) Output tensor — σχηματική αναπαράσταση των 9 καναλιών
axes[4].set_xlim(0, 1)
axes[4].set_ylim(0, 1)
axes[4].set_aspect("equal")
ch_colors = plt.cm.Blues(np.linspace(0.3, 0.8, 9))
for i in range(9):
    rect = plt.Rectangle((0.12, 0.06 + i * 0.095), 0.76, 0.08,
                          facecolor=ch_colors[i], edgecolor="white", lw=0.5)
    axes[4].add_patch(rect)
    axes[4].text(0.5, 0.10 + i * 0.095, f"ch {i+1}",
                 ha="center", va="center", fontsize=7.5, color="white")
axes[4].text(0.5, 0.97, "(5) Tensor", ha="center", va="top",
             fontsize=9, fontweight="bold")
axes[4].text(0.5, 0.02, "X ∈ ℝ⁹ˣ¹²⁸", ha="center", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.2", fc="#E3F2FD", ec="#1565C0", lw=0.8))
axes[4].axis("off")

fig.suptitle("Αγωγός Προ-επεξεργασίας — Raw Signal → CNN Input Tensor",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/eda_preprocessing_pipeline.pdf", bbox_inches="tight")
plt.close()
print("  → eda_preprocessing_pipeline.pdf")


# =============================================================================
# Εκτύπωση στατιστικών για το LaTeX
# =============================================================================
print("\n" + "-" * 50)
print("ΣΤΑΤΙΣΤΙΚΑ ΓΙΑ ΤΗΝ ΔΙΠΛΩΜΑΤΙΚΗ (§3.1–§3.2)")
print("-" * 50)
print(f"Train samples : {len(y_train)} | Test: {len(y_test)}")
print(f"Subjects train: {N_SUBJECTS}")
print()
print("Κατανομή train:")
for c in range(1, 7):
    n = int(np.sum(y_train == c))
    print(f"  {ACTIVITY_NAMES[c]:15s}: {n:4d}  ({100*n/len(y_train):.1f}%)")
print()
print(f"Non-IID JSD²  : mean={jsd.mean():.4f}, std={jsd.std():.4f}, "
      f"min={jsd.min():.4f}, max={jsd.max():.4f}")
print("-" * 50)
print("Done.")
