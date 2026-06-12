# =============================================================================
# ch03_data_analysis / comprehensive_eda.py
# -----------------------------------------------------------------------------
# Πλήρης διερευνητική ανάλυση δεδομένων (EDA) για HAR datasets.
#
# Η ανάλυση γίνεται *agnostic* ως προς την τελική χρήση (δεν προϋποθέτουμε
# FL). Στο τέλος, η §K συνοψίζει ποια ευρήματα επηρεάζουν τη σχεδίαση
# του FL συστήματος.
#
# Ενότητες:
#   A.  Data Quality & Integrity        (shape, NaN/Inf, range, duplicates)
#   B.  Class Distribution               (overall + per-subject + metrics)
#   C.  Descriptive Statistics          (per channel, per class)
#   D.  Distribution & Normality         (histograms, QQ, Shapiro-Wilk)
#   E.  Time-Domain Properties          (waveforms, ACF)
#   F.  Frequency Domain                 (Welch PSD, dominant freq)
#   G.  Channel Correlations             (Pearson + per-class Δ)
#   H.  Dimensionality Reduction        (PCA, t-SNE, silhouette)
#   I.  Outlier Detection (multi-method) (IQR, Z, MAD, IForest, LOF, Mahalan.)
#   J.  Subject Heterogeneity           (JSD², Wasserstein, inter-subj dist.)
#   K.  FL Relevance                     (τι συμπεραίνουμε με FL στο νου)
#
# Τρέχει και στα δύο datasets (UCI HAR + AAL) για σύγκριση.
# Figures → thesis/figures/   (PDF, serif font)
# Reports → implementation/ch03_data_analysis/eda_output/
# =============================================================================

from __future__ import annotations

import os, sys, json, warnings
from pathlib import Path
from collections import Counter
from itertools import groupby

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy import stats as sstats
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon, mahalanobis
from scipy.stats import wasserstein_distance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset_loader import load_uci_har, load_aal, _infer_subject_ids

warnings.filterwarnings("ignore", category=UserWarning)

# --- Paths & matplotlib style ------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
FIG_DIR  = ROOT / "thesis" / "figures"
OUT_DIR  = ROOT / "implementation" / "ch03_data_analysis" / "eda_output"
UCI_DIR  = ROOT / "Datasets" / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"
AAL_DIR  = ROOT / "Datasets" / "aal"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_NAMES = {1: "Walking", 2: "Walk Up", 3: "Walk Down",
                  4: "Sitting", 5: "Standing", 6: "Laying"}
SHORT_LABS = ["WALK", "W-UP", "W-DN", "SIT", "STAND", "LAY"]
DYNAMIC_CLASSES = [1, 2, 3]          # Walking variants
STATIC_CLASSES  = [4, 5, 6]          # Sitting/Standing/Laying
COLORS = ["#2196F3", "#4CAF50", "#8BC34A", "#FF9800", "#F44336", "#9C27B0"]
UCI_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z"]
SAMPLING_RATE = 50   # Hz

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
# Helpers
# =============================================================================

def savefig(name: str):
    """Save to FIG_DIR/<name>.pdf and print path."""
    path = FIG_DIR / f"{name}.pdf"
    plt.savefig(path)
    plt.close()
    print(f"    → {path.relative_to(ROOT)}")


def window_features(X: np.ndarray) -> np.ndarray:
    """Εξαγωγή στατιστικών features ανά παράθυρο.

    Input:  X of shape (N, C, T)  — N παράθυρα, C κανάλια, T δείγματα
    Output: F of shape (N, C × 6) — 6 features ανά κανάλι:
              mean, std, min, max, energy (mean^2), dominant freq
    """
    N, C, T = X.shape
    feats = np.zeros((N, C * 6), dtype=np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0 / SAMPLING_RATE)
    for c in range(C):
        sig = X[:, c, :]                             # (N, T)
        feats[:, c*6 + 0] = sig.mean(axis=1)
        feats[:, c*6 + 1] = sig.std(axis=1)
        feats[:, c*6 + 2] = sig.min(axis=1)
        feats[:, c*6 + 3] = sig.max(axis=1)
        feats[:, c*6 + 4] = (sig ** 2).mean(axis=1)
        # dominant freq: argmax του |FFT| (αγνοώντας DC)
        spectrum = np.abs(np.fft.rfft(sig, axis=1))
        spectrum[:, 0] = 0
        feats[:, c*6 + 5] = freqs[spectrum.argmax(axis=1)]
    return feats


# =============================================================================
# A. Data Quality & Integrity
# =============================================================================

def part_a_quality(X_tr, y_tr, subj_tr, X_te, y_te, name: str) -> dict:
    print(f"\n[A] Data Quality — {name}")
    report = {
        "shape_train": list(X_tr.shape),
        "shape_test":  list(X_te.shape),
        "dtype": str(X_tr.dtype),
        "nan_train": int(np.isnan(X_tr).sum()),
        "inf_train": int(np.isinf(X_tr).sum()),
        "nan_test":  int(np.isnan(X_te).sum()),
        "inf_test":  int(np.isinf(X_te).sum()),
        "value_range": [float(X_tr.min()), float(X_tr.max())],
        "unique_labels": sorted(np.unique(y_tr).tolist()),
        "label_in_range": bool(set(np.unique(y_tr)) <= set(range(1, 7))),
        "n_subjects_train": int(len(np.unique(subj_tr))),
    }
    # duplicates: βρίσκουμε παράθυρα που είναι bitwise identical
    flat = X_tr.reshape(len(X_tr), -1)
    _, idx, counts = np.unique(flat, axis=0, return_index=True, return_counts=True)
    report["n_duplicate_windows"] = int((counts > 1).sum())

    # physical range check (only for accelerometer channels in raw space)
    # after z-score it's not meaningful, so we skip here unless raw
    for k, v in report.items():
        print(f"    {k}: {v}")
    return report


# =============================================================================
# B. Class Distribution
# =============================================================================

def part_b_class_dist(y_tr, y_te, subj_tr, name: str) -> dict:
    print(f"\n[B] Class Distribution — {name}")
    train_counts = np.array([np.sum(y_tr == c) for c in range(1, 7)])
    test_counts  = np.array([np.sum(y_te == c) for c in range(1, 7)])
    p_tr = train_counts / train_counts.sum()

    # imbalance metrics
    entropy   = -np.sum(p_tr * np.log2(p_tr + 1e-12))
    max_ent   = np.log2(6)
    norm_ent  = entropy / max_ent
    imbal_r   = train_counts.max() / train_counts.min()
    gini      = 1 - np.sum(p_tr ** 2)

    report = {
        "train_counts": train_counts.tolist(),
        "test_counts":  test_counts.tolist(),
        "entropy": float(entropy),
        "normalized_entropy": float(norm_ent),
        "imbalance_ratio_max_min": float(imbal_r),
        "gini": float(gini),
    }
    for k, v in report.items():
        print(f"    {k}: {v}")

    # --- Figure: overall class distribution bars ---
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(6)
    w = 0.38
    bars_tr = ax.bar(x - w/2, train_counts, w, label=f"Train (n={train_counts.sum()})",
                     color=COLORS, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_te = ax.bar(x + w/2, test_counts,  w, label=f"Test  (n={test_counts.sum()})",
                     color=COLORS, alpha=0.45, edgecolor="white", linewidth=0.5, hatch="//")
    for bar in list(bars_tr) + list(bars_te):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([ACTIVITY_NAMES[c] for c in range(1, 7)], rotation=15, ha="right")
    ax.set_ylabel("Αριθμός παραθύρων")
    ax.set_xlabel("Κλάση δραστηριότητας")
    ax.set_title(f"Κατανομή κλάσεων — {name}")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(train_counts.max(), test_counts.max()) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(f"eda_class_distribution_{name.lower()}")

    return report


# =============================================================================
# C. Descriptive Statistics per Channel (boxplots per class)
# =============================================================================

def part_c_descriptive(X_tr, y_tr, channels: list[str], name: str) -> dict:
    print(f"\n[C] Descriptive Stats — {name}")
    n_ch = X_tr.shape[1]

    # μέσοι ανά κανάλι, όλα μαζί
    ch_mean = X_tr.mean(axis=(0, 2))
    ch_std  = X_tr.std(axis=(0, 2))
    ch_skew = np.array([sstats.skew(X_tr[:, c, :].ravel()) for c in range(n_ch)])
    ch_kurt = np.array([sstats.kurtosis(X_tr[:, c, :].ravel()) for c in range(n_ch)])

    report = {
        "channel_means": ch_mean.tolist(),
        "channel_stds":  ch_std.tolist(),
        "channel_skewness": ch_skew.tolist(),
        "channel_kurtosis": ch_kurt.tolist(),
    }
    print(f"    per-channel μ range: [{ch_mean.min():.3f}, {ch_mean.max():.3f}]")
    print(f"    per-channel σ range: [{ch_std.min():.3f}, {ch_std.max():.3f}]")

    # --- Figure: boxplot ανά κανάλι ανά κλάση (grouped) ---
    # Για να μη γίνει ακατάστατο, επιλέγουμε 3 κανάλια representative.
    # Στο UCI HAR: body_acc_x (0), body_gyro_x (3), total_acc_z (8)
    repr_idx = [0, 3, 8] if n_ch == 9 else [0]
    repr_names = [channels[i] for i in repr_idx]

    fig, axes = plt.subplots(1, len(repr_idx), figsize=(4.5 * len(repr_idx), 4),
                              sharey=False)
    if len(repr_idx) == 1:
        axes = [axes]
    for ax, ci, cname in zip(axes, repr_idx, repr_names):
        data = [X_tr[y_tr == c, ci, :].ravel() for c in range(1, 7)]
        bp = ax.boxplot(data, labels=SHORT_LABS, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, col in zip(bp["boxes"], COLORS):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_title(cname, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Κατανομή τιμών καναλιών ανά κλάση — {name}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    savefig(f"eda_descriptive_boxplot_{name.lower()}")

    return report


# =============================================================================
# D. Distribution Analysis & Normality Tests
# =============================================================================

def part_d_distribution(X_tr, channels: list[str], name: str) -> dict:
    print(f"\n[D] Distribution & Normality — {name}")
    n_ch = X_tr.shape[1]
    # δειγματοληψία 10k σημείων ανά κανάλι για να τρέξει το Shapiro γρήγορα
    rng = np.random.default_rng(42)
    sw_p = []
    for c in range(n_ch):
        sample = X_tr[:, c, :].ravel()
        if len(sample) > 5000:
            sample = rng.choice(sample, 5000, replace=False)
        # Shapiro-Wilk (null: normal). For large n, it's very sensitive to
        # small deviations, so we also report D'Agostino K²
        _, p_sw = sstats.shapiro(sample)
        sw_p.append(p_sw)
    sw_p = np.array(sw_p)
    report = {
        "shapiro_p_min": float(sw_p.min()),
        "shapiro_p_max": float(sw_p.max()),
        "channels_rejecting_normal_at_0.05": int((sw_p < 0.05).sum()),
    }
    print(f"    Shapiro-Wilk: {report['channels_rejecting_normal_at_0.05']}/{n_ch} "
          f"channels reject normality (α=0.05)")

    # --- Figure: histogram + QQ plot για επιλεγμένα κανάλια ---
    repr_idx = [0, 3, 8] if n_ch == 9 else [0]
    fig, axes = plt.subplots(2, len(repr_idx), figsize=(4.5 * len(repr_idx), 7))
    if len(repr_idx) == 1:
        axes = axes.reshape(-1, 1)
    for col, (ci, cname) in enumerate(zip(repr_idx, [channels[i] for i in repr_idx])):
        data = X_tr[:, ci, :].ravel()
        if len(data) > 20000:
            data = rng.choice(data, 20000, replace=False)

        axes[0, col].hist(data, bins=60, color=COLORS[col % 6], alpha=0.7,
                           edgecolor="white", linewidth=0.3, density=True)
        # superimpose fitted normal
        mu, sg = data.mean(), data.std()
        xs = np.linspace(data.min(), data.max(), 200)
        axes[0, col].plot(xs, sstats.norm.pdf(xs, mu, sg), "k--", lw=1.2,
                           label=f"N({mu:.2f}, {sg:.2f}²)")
        axes[0, col].set_title(f"{cname} (histogram)", fontsize=10)
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(alpha=0.2, linestyle="--")

        sstats.probplot(data, dist="norm", plot=axes[1, col])
        axes[1, col].set_title(f"{cname} (Q-Q plot)", fontsize=10)
        axes[1, col].grid(alpha=0.2, linestyle="--")

    fig.suptitle(f"Κατανομή τιμών & έλεγχος κανονικότητας — {name}",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    savefig(f"eda_distribution_{name.lower()}")

    return report


# =============================================================================
# E. Time-Domain Signal Properties (waveforms + ACF)
# =============================================================================

def part_e_time_domain(X_tr, y_tr, channels: list[str], name: str):
    print(f"\n[E] Time-Domain — {name}")
    n_ch = X_tr.shape[1]
    t = np.arange(X_tr.shape[2]) / SAMPLING_RATE
    # Waveform: ένα παράθυρο ανά κλάση, 3 κανάλια (τα accelerometer)
    repr_idx = [0, 1, 2] if n_ch == 9 else [0]   # body_acc_{x,y,z}

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    for i, c in enumerate(range(1, 7)):
        ax = axes[i]
        idx = np.where(y_tr == c)[0][10]
        for ch_i, col in zip(repr_idx, ["#2196F3", "#F44336", "#4CAF50"]):
            ax.plot(t, X_tr[idx, ch_i, :], color=col, lw=1.0, alpha=0.9,
                    label=channels[ch_i] if i == 0 else None)
        ax.set_title(ACTIVITY_NAMES[c], fontweight="bold", pad=4)
        ax.axhline(0, color="gray", lw=0.4, linestyle="--")
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if i >= 4: ax.set_xlabel("Χρόνος (s)")
        if i % 2 == 0: ax.set_ylabel("Normalized amplitude")
    fig.legend(loc="lower center", ncol=3, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Χρονοσειρές ανά κλάση — {name}", fontsize=12, y=1.01)
    plt.tight_layout()
    savefig(f"eda_signals_per_activity_{name.lower()}")

    # ACF plot — ένα channel, όλες οι κλάσεις
    max_lag = 60
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ci = 0  # body_acc_x
    for c, col in zip(range(1, 7), COLORS):
        # mean ACF across windows of this class (sampled)
        class_idx = np.where(y_tr == c)[0]
        if len(class_idx) > 100:
            class_idx = np.random.default_rng(42).choice(class_idx, 100, replace=False)
        acfs = []
        for idx in class_idx:
            s = X_tr[idx, ci, :]
            s = s - s.mean()
            denom = np.dot(s, s) + 1e-12
            acf = np.correlate(s, s, mode="full")[len(s)-1:] / denom
            acfs.append(acf[:max_lag])
        mean_acf = np.mean(acfs, axis=0)
        ax.plot(np.arange(max_lag) / SAMPLING_RATE, mean_acf, color=col,
                lw=1.3, label=ACTIVITY_NAMES[c])
    ax.axhline(0, color="gray", lw=0.4, linestyle="--")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel(f"ACF ({channels[ci]})")
    ax.set_title(f"Autocorrelation ανά κλάση — {name}")
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(f"eda_acf_{name.lower()}")


# =============================================================================
# F. Frequency Domain (Welch PSD + dominant frequency)
# =============================================================================

def part_f_frequency(X_tr, y_tr, channels: list[str], name: str) -> dict:
    print(f"\n[F] Frequency Domain — {name}")
    n_ch = X_tr.shape[1]
    ci = 0   # body_acc_x
    T = X_tr.shape[2]
    nperseg = min(T, 64)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    dom_freqs = {}
    for c, col in zip(range(1, 7), COLORS):
        idx = np.where(y_tr == c)[0]
        if len(idx) > 200:
            idx = np.random.default_rng(42).choice(idx, 200, replace=False)
        psds = []
        for i in idx:
            f, Pxx = welch(X_tr[i, ci, :], fs=SAMPLING_RATE, nperseg=nperseg)
            psds.append(Pxx)
        mean_psd = np.mean(psds, axis=0)
        # dominant freq, excluding DC
        mask = f > 0.2
        dom_freqs[c] = float(f[mask][mean_psd[mask].argmax()])
        ax.semilogy(f, mean_psd, color=col, lw=1.3,
                    label=f"{ACTIVITY_NAMES[c]} ({dom_freqs[c]:.2f} Hz)")
    ax.set_xlabel("Συχνότητα (Hz)")
    ax.set_ylabel(f"PSD ({channels[ci]})")
    ax.set_title(f"Power Spectral Density ανά κλάση — {name} (Welch)")
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    ax.grid(which="both", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(f"eda_psd_{name.lower()}")

    print(f"    Dominant freqs (body_acc_x): {dom_freqs}")
    return {"dominant_frequencies_Hz": dom_freqs}


# =============================================================================
# G. Channel Correlation
# =============================================================================

def part_g_correlation(X_tr, y_tr, channels: list[str], name: str):
    print(f"\n[G] Channel Correlations — {name}")
    n_ch = X_tr.shape[1]
    if n_ch < 2:
        print("    (skipped: single-channel dataset)")
        return
    # correlation matrix across all windows: mean value per (window, channel)
    feat = X_tr.mean(axis=2)   # (N, C)
    corr_all = np.corrcoef(feat.T)

    # per-group (dynamic vs static)
    dyn_mask = np.isin(y_tr, DYNAMIC_CLASSES)
    stat_mask = np.isin(y_tr, STATIC_CLASSES)
    corr_dyn  = np.corrcoef(feat[dyn_mask].T)
    corr_stat = np.corrcoef(feat[stat_mask].T)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, M, title in zip(
        axes,
        [corr_all, corr_dyn, corr_stat],
        [f"Όλες οι κλάσεις", "Δυναμικές", "Στατικές"]
    ):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n_ch)); ax.set_yticks(range(n_ch))
        ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(channels, fontsize=7)
        ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=axes, fraction=0.022, pad=0.02, label="Pearson r")
    fig.suptitle(f"Συσχετίσεις καναλιών — {name}", fontsize=12, y=1.02)
    savefig(f"eda_correlation_{name.lower()}")


# =============================================================================
# H. Dimensionality Reduction (PCA + t-SNE + silhouette)
# =============================================================================

def part_h_dimred(X_tr, y_tr, name: str) -> dict:
    print(f"\n[H] Dimensionality Reduction — {name}")
    # Για single-channel datasets (π.χ. AAL pre-extracted features),
    # δουλεύουμε κατευθείαν στον feature vector. Για multi-channel raw
    # σήματα, εξάγουμε stats per channel.
    if X_tr.shape[1] == 1:
        F = X_tr.reshape(len(X_tr), -1)
    else:
        F = window_features(X_tr)

    n_comp = min(10, F.shape[1], F.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    F_pca = pca.fit_transform(F)

    sil_pca = silhouette_score(F_pca[:, :2], y_tr, sample_size=2000,
                                random_state=42)

    # t-SNE (on subset for speed)
    rng = np.random.default_rng(42)
    sub = rng.choice(len(F), size=min(3000, len(F)), replace=False)
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=42).fit_transform(F_pca[sub, :min(n_comp, 10)])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # (1) Variance explained
    cum = np.cumsum(pca.explained_variance_ratio_)
    xs = range(1, n_comp + 1)
    axes[0].bar(xs, pca.explained_variance_ratio_, color="#42A5F5",
                 edgecolor="white", alpha=0.85)
    axes[0].plot(xs, cum, "o-", color="#E53935",
                 label="Αθροιστική", lw=1.5)
    axes[0].set_xlabel("Κύρια Συνιστώσα")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title(f"PCA scree (top {n_comp})")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # (2) PCA scatter
    for c, col in zip(range(1, 7), COLORS):
        mask = y_tr == c
        axes[1].scatter(F_pca[mask, 0], F_pca[mask, 1], s=6, alpha=0.35,
                         color=col, label=ACTIVITY_NAMES[c])
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title(f"PCA — silhouette={sil_pca:.3f}")
    axes[1].legend(fontsize=8, markerscale=2.5, framealpha=0.9, ncol=2)
    axes[1].grid(alpha=0.2, linestyle="--")

    # (3) t-SNE
    for c, col in zip(range(1, 7), COLORS):
        mask = y_tr[sub] == c
        axes[2].scatter(ts[mask, 0], ts[mask, 1], s=6, alpha=0.55,
                         color=col, label=ACTIVITY_NAMES[c])
    axes[2].set_xlabel("t-SNE-1"); axes[2].set_ylabel("t-SNE-2")
    axes[2].set_title(f"t-SNE (n={len(sub)})")
    axes[2].grid(alpha=0.2, linestyle="--")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"Διαχωρισιμότητα κλάσεων — {name}", fontsize=12, y=1.02)
    plt.tight_layout()
    savefig(f"eda_dimred_{name.lower()}")

    report = {
        "pca_variance_top2": float(cum[min(1, n_comp-1)]),
        "pca_variance_top5": float(cum[min(4, n_comp-1)]),
        "silhouette_pca2": float(sil_pca),
    }
    for k, v in report.items():
        print(f"    {k}: {v}")
    return report


# =============================================================================
# I. Outlier Detection (Multi-method)
# =============================================================================

def part_i_outliers(X_tr, y_tr, name: str) -> dict:
    print(f"\n[I] Outlier Detection (multi-method) — {name}")

    # Όλες οι μέθοδοι τρέχουν σε window-level features (αυτά που δίνουν
    # νόημα στη συγκεκριμένη αναπαράσταση), ώστε να είναι συγκρίσιμες.
    # Τα univariate IQR/Z-score εφαρμόζονται σε κάθε feature ξεχωριστά και
    # marking γίνεται όταν πάνω από ~10% των features βγαίνουν outlier —
    # αποφεύγουμε το "any-channel" που είναι τετριμμένα θετικό σε υψηλή διάσταση.
    if X_tr.shape[1] == 1:
        F = X_tr.reshape(len(X_tr), -1)
    else:
        F = window_features(X_tr)
    F_std = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
    N, D = F_std.shape
    per_feat_thr = max(1, int(0.10 * D))   # flag if ≥10% of features are outliers

    z = np.abs(F_std)
    zsc_mask = (z > 3).sum(axis=1) >= per_feat_thr

    med = np.median(F_std, axis=0)
    mad = np.median(np.abs(F_std - med), axis=0) + 1e-9
    mz  = np.abs(0.6745 * (F_std - med) / mad)
    mad_mask = (mz > 3.5).sum(axis=1) >= per_feat_thr

    q1, q3 = np.percentile(F_std, [25, 75], axis=0)
    iqr_span = q3 - q1
    iqr_lo, iqr_hi = q1 - 1.5 * iqr_span, q3 + 1.5 * iqr_span
    iqr_mask = ((F_std < iqr_lo) | (F_std > iqr_hi)).sum(axis=1) >= per_feat_thr

    iso = IsolationForest(contamination=0.05, random_state=42).fit(F_std)
    iso_mask = iso.predict(F_std) == -1

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_mask = lof.fit_predict(F_std) == -1

    # Mahalanobis distance (use shrinkage-friendly covariance)
    cov = np.cov(F_std, rowvar=False) + 1e-3 * np.eye(F_std.shape[1])
    icov = np.linalg.pinv(cov)
    mu = F_std.mean(axis=0)
    d2 = np.array([mahalanobis(f, mu, icov) ** 2 for f in F_std])
    # threshold: chi² with df=n_features, α=0.01
    thr = sstats.chi2.ppf(0.99, F_std.shape[1])
    mah_mask = d2 > thr

    methods = {
        "IQR (any-channel)":     iqr_mask,
        "Z-score > 3":           zsc_mask,
        "Mod. Z (MAD) > 3.5":    mad_mask,
        "Isolation Forest":      iso_mask,
        "LOF (k=20)":            lof_mask,
        "Mahalanobis (χ² 0.99)": mah_mask,
    }

    # --- Agreement analysis ---
    names_list = list(methods.keys())
    n_meth = len(names_list)
    jaccard = np.zeros((n_meth, n_meth))
    for i in range(n_meth):
        for j in range(n_meth):
            a, b = methods[names_list[i]], methods[names_list[j]]
            inter = (a & b).sum()
            union = (a | b).sum()
            jaccard[i, j] = inter / union if union > 0 else 1.0

    # Consensus: παράθυρο flagged by ≥ 2 methods (out of 6)
    vote = np.sum([m.astype(int) for m in methods.values()], axis=0)
    consensus = vote >= 2

    report = {
        "rates": {k: float(v.mean()) for k, v in methods.items()},
        "consensus_rate_ge2": float(consensus.mean()),
        "consensus_n": int(consensus.sum()),
    }
    for k, v in report["rates"].items():
        print(f"    {k:30s} flags: {v*100:5.2f}%")
    print(f"    Consensus (≥2 methods): {report['consensus_rate_ge2']*100:.2f}% "
          f"({report['consensus_n']} παράθυρα)")

    # --- Figure: outlier rates + jaccard heatmap + vote histogram ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    rates = np.array([v.mean() for v in methods.values()]) * 100
    axes[0].barh(names_list, rates, color="#42A5F5", edgecolor="white")
    for i, r in enumerate(rates):
        axes[0].text(r + 0.1, i, f"{r:.2f}%", va="center", fontsize=9)
    axes[0].set_xlabel("% outliers flagged")
    axes[0].set_title("Ποσοστό outliers ανά μέθοδο")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(axis="x", alpha=0.3, linestyle="--")

    im = axes[1].imshow(jaccard, cmap="YlGnBu", vmin=0, vmax=1)
    axes[1].set_xticks(range(n_meth)); axes[1].set_yticks(range(n_meth))
    axes[1].set_xticklabels(names_list, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(names_list, fontsize=8)
    axes[1].set_title("Jaccard similarity μεταξύ μεθόδων")
    for i in range(n_meth):
        for j in range(n_meth):
            axes[1].text(j, i, f"{jaccard[i,j]:.2f}", ha="center", va="center",
                          fontsize=7, color="black" if jaccard[i,j] < 0.6 else "white")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Vote histogram
    votes, counts = np.unique(vote, return_counts=True)
    bar_colors = ["#66BB6A" if v < 2 else "#EF5350" for v in votes]
    axes[2].bar(votes, counts, color=bar_colors, edgecolor="white")
    for v, c in zip(votes, counts):
        axes[2].text(v, c, str(c), ha="center", va="bottom", fontsize=9)
    axes[2].axvline(1.5, color="black", lw=1, linestyle="--")
    axes[2].set_xlabel("# μεθόδων που σημαδεύουν το παράθυρο")
    axes[2].set_ylabel("Αριθμός παραθύρων")
    axes[2].set_title("Κατανομή ψήφων (consensus ≥ 2)")
    axes[2].set_xticks(range(n_meth + 1))
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"Outlier Detection (multi-method) — {name}", fontsize=12, y=1.03)
    plt.tight_layout()
    savefig(f"eda_outliers_{name.lower()}")

    report["jaccard_matrix"] = jaccard.tolist()
    report["method_order"] = names_list
    return report


# =============================================================================
# J. Subject Heterogeneity (JSD² + Wasserstein on features)
# =============================================================================

def part_j_subject_hetero(X_tr, y_tr, subj_tr, name: str) -> dict:
    print(f"\n[J] Subject Heterogeneity — {name}")
    subjects = np.unique(subj_tr)
    N_S = len(subjects)

    # (1) class distribution per subject + JSD²
    dist = np.zeros((N_S, 6))
    for i, s in enumerate(subjects):
        mask = subj_tr == s
        counts = np.array([np.sum(y_tr[mask] == c) for c in range(1, 7)])
        total  = counts.sum()
        dist[i] = counts / total if total > 0 else counts
    global_dist = np.array([np.sum(y_tr == c) for c in range(1, 7)], float)
    global_dist /= global_dist.sum()
    jsd = np.array([jensenshannon(dist[i], global_dist) ** 2 for i in range(N_S)])

    # (2) feature-space distance: mean Wasserstein per channel between each
    #     subject and the pooled distribution (άλλος τρόπος ετερογένειας)
    feat = X_tr.mean(axis=2)   # (N, C), per-window mean per channel
    pooled = feat
    w_dist = np.zeros(N_S)
    for i, s in enumerate(subjects):
        mask = subj_tr == s
        if mask.sum() < 5:
            w_dist[i] = np.nan
            continue
        per_ch = [wasserstein_distance(feat[mask, ch], pooled[:, ch])
                  for ch in range(feat.shape[1])]
        w_dist[i] = float(np.mean(per_ch))

    # --- Figure: heatmap + JSD bars + Wasserstein bars ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                              gridspec_kw={"width_ratios": [3, 1, 1]})
    im = axes[0].imshow(dist, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.4)
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels(SHORT_LABS, fontsize=9)
    axes[0].set_yticks(range(N_S))
    axes[0].set_yticklabels([f"S{int(s):02d}" for s in subjects], fontsize=7)
    axes[0].set_xlabel("Κλάση")
    axes[0].set_ylabel("Subject")
    axes[0].set_title("Κατανομή κλάσεων ανά subject")
    plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, label="Ratio")

    cols1 = ["#E53935" if j > 0.05 else "#42A5F5" for j in jsd]
    axes[1].barh(range(N_S), jsd, color=cols1, alpha=0.85)
    axes[1].axvline(jsd.mean(), color="black", lw=1.2, ls="--",
                     label=f"μ={jsd.mean():.3f}")
    axes[1].set_yticks(range(N_S))
    axes[1].set_yticklabels([f"S{int(s):02d}" for s in subjects], fontsize=7)
    axes[1].set_xlabel("JSD² (label dist.)")
    axes[1].set_title("Ετερογένεια labels")
    axes[1].legend(fontsize=8); axes[1].grid(axis="x", alpha=0.3, ls="--")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    valid = ~np.isnan(w_dist)
    cols2 = ["#E53935" if w > np.nanmedian(w_dist) else "#42A5F5" for w in w_dist]
    axes[2].barh(range(N_S), np.nan_to_num(w_dist), color=cols2, alpha=0.85)
    axes[2].axvline(np.nanmean(w_dist), color="black", lw=1.2, ls="--",
                     label=f"μ={np.nanmean(w_dist):.3f}")
    axes[2].set_yticks(range(N_S))
    axes[2].set_yticklabels([f"S{int(s):02d}" for s in subjects], fontsize=7)
    axes[2].set_xlabel("Avg Wasserstein (features)")
    axes[2].set_title("Ετερογένεια features")
    axes[2].legend(fontsize=8); axes[2].grid(axis="x", alpha=0.3, ls="--")
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

    fig.suptitle(f"Ετερογένεια χρηστών — {name}", fontsize=12, y=1.02)
    plt.tight_layout()
    savefig(f"eda_subject_heterogeneity_{name.lower()}")

    report = {
        "n_subjects": N_S,
        "jsd_mean": float(jsd.mean()),
        "jsd_std":  float(jsd.std()),
        "jsd_min":  float(jsd.min()),
        "jsd_max":  float(jsd.max()),
        "jsd_pct_above_0.05": float((jsd > 0.05).mean()),
        "wasserstein_mean": float(np.nanmean(w_dist)),
        "wasserstein_std":  float(np.nanstd(w_dist)),
    }
    for k, v in report.items():
        print(f"    {k}: {v}")
    return report


# =============================================================================
# Driver
# =============================================================================

def run_for_dataset(X_tr, y_tr, subj_tr, X_te, y_te, channels, name):
    print(f"\n{'='*70}\nEDA: {name}\n{'='*70}")
    full_report = {"name": name}
    full_report["A_quality"]      = part_a_quality(X_tr, y_tr, subj_tr, X_te, y_te, name)
    full_report["B_class_dist"]   = part_b_class_dist(y_tr, y_te, subj_tr, name)
    full_report["C_descriptive"]  = part_c_descriptive(X_tr, y_tr, channels, name)
    full_report["D_distribution"] = part_d_distribution(X_tr, channels, name)
    part_e_time_domain(X_tr, y_tr, channels, name)
    full_report["F_frequency"]    = part_f_frequency(X_tr, y_tr, channels, name)
    part_g_correlation(X_tr, y_tr, channels, name)
    full_report["H_dimred"]       = part_h_dimred(X_tr, y_tr, name)
    full_report["I_outliers"]     = part_i_outliers(X_tr, y_tr, name)
    full_report["J_heterogeneity"]= part_j_subject_hetero(X_tr, y_tr, subj_tr, name)
    return full_report


def main():
    reports = {}

    # --- UCI HAR (raw inertial, 9 channels × 128 samples) --------------------
    tr_u, te_u, stats_u = load_uci_har(UCI_DIR, normalize=True)
    X_tr_u  = tr_u.X.numpy()
    y_tr_u  = (tr_u.y.numpy() + 1)      # 1-indexed για consistency
    X_te_u  = te_u.X.numpy()
    y_te_u  = (te_u.y.numpy() + 1)
    subj_u  = tr_u.subject_ids
    reports["UCI_HAR"] = run_for_dataset(X_tr_u, y_tr_u, subj_u,
                                         X_te_u, y_te_u,
                                         UCI_CHANNELS, "UCI_HAR")

    # --- AAL (561 features → reshape as (N, 1, 561)) ---------------
    # Για την AAL, τα features είναι προ-εξαγμένα → treat ως (N, 1, 561)
    tr_a, te_a, stats_a = load_aal(AAL_DIR, normalize=True)
    X_tr_a  = tr_a.X.numpy()
    y_tr_a  = (tr_a.y.numpy() + 1)
    X_te_a  = te_a.X.numpy()
    y_te_a  = (te_a.y.numpy() + 1)
    subj_a  = tr_a.subject_ids
    reports["AAL"] = run_for_dataset(X_tr_a, y_tr_a, subj_a,
                                      X_te_a, y_te_a,
                                      ["feature_vector"], "AAL")

    # --- Save reports --------------------------------------------------------
    def json_safe(obj):
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(x) for x in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUT_DIR / "eda_report.json", "w") as f:
        json.dump(json_safe(reports), f, indent=2, ensure_ascii=False)
    print(f"\n✔ Full report → {(OUT_DIR/'eda_report.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
