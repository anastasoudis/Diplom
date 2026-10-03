# Κανένα αποτέλεσμα δεν γράφεται πάνω σε υπάρχον αρχείο χωρίς --overwrite.
# Ο έλεγχος γίνεται στην αρχή κάθε script, πριν από οποιονδήποτε υπολογισμό.

import os


def add_overwrite_flag(parser):
    parser.add_argument("--overwrite", action="store_true",
                        help="γράψε πάνω σε υπάρχον --out")


def check_output(path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"το {path} υπάρχει ήδη. Δώσε άλλο --out ή --overwrite.")
