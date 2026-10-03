# Ορίσματα της κεντρικής εκπαίδευσης και των κλασικών μοντέλων σύγκρισης.

import argparse

from implementation.common.outputs import add_overwrite_flag


def centralized_args(argv=None):
    p = argparse.ArgumentParser(description="Κεντρική εκπαίδευση στο UCI HAR.")
    # 20 seeds (0-19), ώστε οι διαφορές μεταξύ τοπολογιών να ξεχωρίζουν από τη
    # διακύμανση μεταξύ τρεξιμάτων
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_name", type=str, default="ann", choices=["ann", "logreg"])
    # στις 100 εποχές και η κεντρική και η ομοσπονδιακή εκπαίδευση έχουν σταθεροποιηθεί
    p.add_argument("--epochs", type=int, default=100)
    # Εποχές στις οποίες μετριέται και το test (καμπύλη εκπαίδευσης). Αλλάζουν τη
    # συνέχεια της γεννήτριας, οπότε για ίδια αποτελέσματα πρέπει να είναι ίδιες.
    p.add_argument("--checkpoints", type=int, nargs="+", default=[10, 25, 50, 100])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--criterion", type=str, default="cross_entropy", choices=["cross_entropy"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--out", type=str, default=None,
                   help="προεπιλογή results/centralized/har_central_<model>.json")
    add_overwrite_flag(p)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = f"results/centralized/har_central_{args.model_name}.json"
    return args


def classic_args(argv=None):
    p = argparse.ArgumentParser(description="Κλασικά μοντέλα του scikit-learn, με τις προεπιλογές τους.")
    # τα ίδια seeds με το δίκτυο ακόμη και για τα ντετερμινιστικά μοντέλα, ώστε να
    # φαίνεται ότι δεν εξαρτώνται από το seed
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--models", type=str, nargs="+",
                   default=["dummy", "linear_svm", "rbf_svm", "random_forest"])
    p.add_argument("--out", type=str, default="results/centralized/har_central_classic.json")
    add_overwrite_flag(p)
    return p.parse_args(argv)
