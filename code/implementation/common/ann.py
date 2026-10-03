# Το δίκτυο της εργασίας. Το ίδιο αρχείο το χρησιμοποιούν η κεντρική, η
# ομοσπονδιακή και η ατομική εκπαίδευση, ώστε οι συγκρίσεις τους να μην
# επηρεάζονται από διαφορετικά αντίγραφα της αρχιτεκτονικής.
#
# Πλήρως συνδεδεμένο και όχι συνελικτικό, γιατί η είσοδος είναι 561
# χαρακτηριστικά χωρίς χρονική ή χωρική διάταξη μεταξύ τους.

import torch
import torch.nn as nn

N_FEATURES = 561
N_CLASSES = 6


class ANN(nn.Module):
    """561 -> 256 -> 128 -> 6, με ReLU και dropout μετά από κάθε κρυφό επίπεδο."""

    def __init__(self, in_features: int = N_FEATURES, num_classes: int = N_CLASSES,
                 hidden=(256, 128), dropout: float = 0.3) -> None:
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # δέχεται (N, 1, 561) όπως τα δίνει το HARDataset, ή (N, 561)
        return self.net(input_tensor.flatten(start_dim=1))
