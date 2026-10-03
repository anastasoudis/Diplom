# Πολυωνυμική logistic regression, για σύγκριση με το δίκτυο στην κεντρική εκπαίδευση.
#
# Είναι το ANN χωρίς τα κρυφά επίπεδα. Γράφεται σε PyTorch ώστε να εκπαιδεύεται στον
# ίδιο βρόχο με τις ίδιες ρυθμίσεις (Adam 1e-3, batch 64, 100 εποχές, ίδια seeds) και
# η σύγκριση να αλλάζει μόνο ένα πράγμα. Δεν έχει dropout, αφού δεν υπάρχει κρυφό επίπεδο.

import torch
import torch.nn as nn

from implementation.common.ann import N_CLASSES, N_FEATURES


class LogReg(nn.Module):
    """Linear(561 -> 6). Η softmax περιέχεται στην cross-entropy, όπως και στο ANN."""

    def __init__(self, in_features: int = N_FEATURES, num_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.linear(input_tensor.flatten(start_dim=1))
