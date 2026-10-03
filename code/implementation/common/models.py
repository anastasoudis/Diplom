# Επιλογή αρχιτεκτονικής από το όνομά της (--model_name).

from implementation.common.ann import ANN
from implementation.common.logreg import LogReg

# dropout: αν η αρχιτεκτονική έχει θέση για dropout. Αν όχι, το build() δεν της το περνά.
MODELS = {"ann": {"cls": ANN, "dropout": True, "repr": "ANN(561->256->128->6)"},
          "logreg": {"cls": LogReg, "dropout": False, "repr": "LogReg(561->6)"}}


class ModelBuilder:
    def __init__(self, model_name: str = "ann"):
        if model_name not in MODELS:
            raise NotImplementedError(f"άγνωστη αρχιτεκτονική {model_name!r}, διαθέσιμες {sorted(MODELS)}")
        self.name = model_name
        self.spec = MODELS[model_name]

    def __repr__(self):
        return self.spec["repr"]

    @property
    def uses_dropout(self) -> bool:
        return self.spec["dropout"]

    def build(self, dropout: float):
        """Νέο μοντέλο. Η αρχικοποίηση των βαρών καταναλώνει την καθολική γεννήτρια του torch."""
        return self.spec["cls"](**({"dropout": dropout} if self.uses_dropout else {}))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
