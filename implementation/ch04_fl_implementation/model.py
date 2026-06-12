# =============================================================================
# ch04_fl_implementation / model.py
# -----------------------------------------------------------------------------
# 1D-CNN για ταξινόμηση δραστηριότητας (HAR) από προ-εξαχθέντα features.
# Είσοδος: [batch, 1, 561] — 1 κανάλι × 561 features ανά παράθυρο
# Έξοδος: [batch, 6]       — logits για 6 κλάσεις (softmax στο loss)
#
# Αντιστοιχεί σε: §4.1 (Σχεδιασμός Νευρωνικού Δικτύου) της διπλωματικής
# =============================================================================

import torch
import torch.nn as nn


# Hyperparameters αρχιτεκτονικής — όλες σε ένα σημείο για εύκολη αλλαγή
N_CHANNELS  = 1    # κανάλι εισόδου (τα 561 features ως 1D ακολουθία)
N_CLASSES   = 6    # WALKING, W_UP, W_DOWN, SITTING, STANDING, LAYING
WIN_SIZE    = 561  # μήκος ακολουθίας = 561 προ-εξαχθέντα features ανά παράθυρο

CONV1_FILTERS = 64   # πόσα patterns ψάχνει το πρώτο conv layer
CONV2_FILTERS = 128  # το δεύτερο layer μαθαίνει πιο σύνθετα patterns
KERNEL_SIZE   = 9    # "εύρος" κάθε φίλτρου σε χρονικά βήματα (~0.18 s @ 50 Hz)
POOL_SIZE     = 2    # MaxPool μειώνει κατά 2 τη χρονική διάσταση

FC_HIDDEN   = 128   # νευρώνες στο κρυφό fully-connected layer
DROPOUT_P   = 0.5   # πιθανότητα dropout — μειώνει overfitting στο FL


class HAR_CNN(nn.Module):
    """1D-CNN για αναγνώριση δραστηριότητας από προ-εξαχθέντα features.

    Αρχιτεκτονική:
        Conv1d(1→64, k=9) → BatchNorm → ReLU → MaxPool(2)
        Conv1d(64→128, k=9) → BatchNorm → ReLU → MaxPool(2)
        Flatten → Dropout(0.5) → Linear(→128) → ReLU → Linear(6)

    Είσοδος: [B, 1, 561] — 1 κανάλι × 561 features (time+frequency domain)
    Έξοδος:  [B, 6]      — logits ανά κλάση δραστηριότητας

    Το μέγεθος της FC εισόδου υπολογίζεται αυτόματα με _get_fc_input_size.
    """

    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                 win_size=WIN_SIZE):
        super().__init__()

        # --- Convolutional blocks -------------------------------------------
        # Κάθε block: Conv1d → BatchNorm → ReLU → MaxPool
        # Το BatchNorm σταθεροποιεί την εκπαίδευση — κρίσιμο στο FL όπου
        # τα batches κάθε client μπορεί να έχουν πολύ διαφορετικές κατανομές.

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, CONV1_FILTERS, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2),
            nn.BatchNorm1d(CONV1_FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2),
            nn.BatchNorm1d(CONV2_FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
        )

        # --- Fully connected head -------------------------------------------
        # Ο αριθμός εισόδων στο FC εξαρτάται από το win_size και το pooling.
        # Αντί να τον hardcode, τον υπολογίζουμε δυναμικά με ένα dummy forward pass.
        fc_in = self._get_fc_input_size(n_channels, win_size)

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_P),          # απενεργοποίηση τυχαίων νευρώνων κατά training
            nn.Linear(fc_in, FC_HIDDEN),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN, n_classes),  # έξοδος: 6 raw scores (logits)
        )

    def _get_fc_input_size(self, n_channels: int, win_size: int) -> int:
        """Υπολογίζει το μέγεθος εξόδου των conv layers με dummy forward pass.

        Αποφεύγουμε hardcoded τιμές που θα έσπαγαν αν αλλάξουν kernel/pool params.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, win_size)  # ένα fake batch
            out   = self.conv2(self.conv1(dummy))
            return out.view(1, -1).shape[1]  # flatten και μέτρηση dimensions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor [batch, 1, 561]
        Returns:
            logits: Tensor [batch, 6] — raw scores πριν το softmax
        """
        x = self.conv1(x)          # [B, 1, 561] → [B, 64, 280]
        x = self.conv2(x)          # [B, 64, 280] → [B, 128, 140]
        x = x.view(x.size(0), -1)  # flatten: [B, 128, 140] → [B, 17920]
        return self.classifier(x)  # [B, 17920] → [B, 6]


def get_model(device: str = "cpu") -> HAR_CNN:
    """Δημιουργεί και επιστρέφει το μοντέλο στο επιθυμητό device."""
    model = HAR_CNN()
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Μετράει τις εκπαιδεύσιμες παραμέτρους — χρήσιμο για αναφορά στη διπλωματική."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
