# Diplom — Federated Learning για HAR με Μηχανισμούς Κινήτρων (AAL)

Διπλωματική εργασία — Τμήμα Ηλεκτρολόγων Μηχανικών & Μηχανικών Υπολογιστών, Δ.Π.Θ.

Σχεδιασμός και υλοποίηση συστήματος **Ομοσπονδιακής Μάθησης (Federated Learning)** για
**Αναγνώριση Ανθρώπινης Δραστηριότητας (HAR)** σε περιβάλλοντα **Υποβοηθούμενης Διαβίωσης
(Ambient Assisted Living, AAL)**, ενισχυμένου με **μηχανισμό κινήτρων** βασισμένο σε Θεωρία
Παιγνίων (Stackelberg) και σε αποτίμηση συνεισφοράς (Shapley value / Leave-One-Out).

> **Κατάσταση:** Πρόοδος έως τον σχεδιασμό του Μηχανισμού Κινήτρων (Κεφ. 1–4).
> Η πειραματική αξιολόγηση (Κεφ. 5–6) ακολουθεί σε επόμενη iteration.

## Δομή

| Φάκελος | Περιεχόμενο |
|---|---|
| `thesis/` | Το κείμενο της διπλωματικής έως το σημείο αυτό: `thesis_ch1-4.pdf` + πηγές LaTeX (`src/`, `figures/`) |
| `implementation/` | Η υλοποίηση σε Python / PyTorch |
| `config/` | `requirements.txt` |

### `implementation/`
- `ch03_data_analysis/` — φόρτωση δεδομένων (UCI HAR, AAL) & exploratory data analysis (EDA)
- `ch04_fl_implementation/` — μοντέλο 1D-CNN, FL client/server (custom FedAvg σε PyTorch), μηχανισμός κινήτρων (Stackelberg)

## Εκτέλεση

```bash
pip install -r config/requirements.txt
# Τοποθέτησε τα datasets (δες παρακάτω) στον φάκελο Datasets/
```

- **Ανάλυση δεδομένων (Κεφ. 3):** `implementation/ch03_data_analysis/` — φόρτωση datasets, EDA, παραγωγή figures.
- **FL σύστημα (Κεφ. 4):** `implementation/ch04_fl_implementation/` — μοντέλο 1D-CNN, client/server (FedAvg) και μηχανισμός κινήτρων (Stackelberg), ως modules.

> Η πλήρης πειραματική αξιολόγηση (Κεφ. 5–6) ακολουθεί σε επόμενη iteration.

## Datasets

Δεν περιλαμβάνονται στο repo (μέγεθος / άδειες τρίτων). Κατέβασέ τα και τοποθέτησέ τα ως εξής:

- **AAL (κύριο) — Smartphone Dataset for HAR in AAL** (Davis & Owusu) —
  <https://archive.ics.uci.edu/dataset/364/smartphone+dataset+for+human+activity+recognition+har+in+ambient+assisted+living+aal>
  → στον φάκελο **`Datasets/aal/`**
- **UCI HAR** (baseline αναφοράς) —
  <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
  → στον φάκελο **`Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/`**

## Το κείμενο (PDF)

Έτοιμο για ανάγνωση: **[`thesis/thesis_ch1-4.pdf`](thesis/thesis_ch1-4.pdf)** (committed στο repo).

Προαιρετικό rebuild από τις πηγές LaTeX (απαιτεί XeLaTeX + biber):

```bash
cd thesis && make all    # → build/thesis_ch1-4.pdf  (το build/ δεν ανεβαίνει στο git)
```

## Άδεια

[MIT](LICENSE) © 2026 Dimitrios Anastasoudis
