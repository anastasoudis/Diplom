# Τρέχει με τη σειρά όλη την προετοιμασία των δεδομένων, δηλαδή τον έλεγχο στα ωμά αρχεία,
# τη διερευνητική ανάλυση, την εξαγωγή των επεξεργασμένων αρχείων και τον έλεγχο κλίμακας.
#
#   python -m implementation.dataset.run_eda_uci

import sys
import time

from implementation.dataset import eda_uci, export_processed, inspect_raw_uci, scaling_check

STAGES = [("έλεγχος στα ωμά αρχεία", inspect_raw_uci.main),
          ("διερευνητική ανάλυση", eda_uci.main),
          ("εξαγωγή στο data/har_processed", lambda: export_processed.main([])),
          ("κλίμακα πριν και μετά τη z-score", scaling_check.main)]


def main():
    t0 = time.time()
    for i, (name, fn) in enumerate(STAGES, 1):
        print(f"\n=== {i}/{len(STAGES)} {name} ===")
        rc = fn()
        if rc:
            print(f"το στάδιο {i} απέτυχε, διακοπή")
            return rc
    print(f"\nολοκληρώθηκε σε {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
