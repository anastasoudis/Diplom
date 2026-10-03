# Αποτίμηση της συνεισφοράς χρηστών στην ομοσπονδιακή μάθηση

Ο κώδικας της διπλωματικής μου εργασίας. Με αυτόν βγήκαν όλα τα αποτελέσματα του κειμένου, από τον
έλεγχο των δεδομένων και τη σύγκριση κεντρικής, ομοσπονδιακής και ατομικής εκπαίδευσης ως την αποτίμηση
της συνεισφοράς κάθε χρήστη με τιμές Shapley.

## Τι υπολογίζει

Τα δεδομένα είναι το UCI HAR (Anguita κ.ά. 2013), με 10.299 samples από τους αισθητήρες ενός κινητού
τηλεφώνου που φορούσαν 30 εθελοντές. Κάθε sample έχει 561 χαρακτηριστικά και μία από έξι δραστηριότητες
(περπάτημα, ανέβασμα και κατέβασμα σκάλας, καθιστή, όρθια και ξαπλωτή θέση). Οι 21 εθελοντές του train
είναι οι clients της ομοσπονδιακής μάθησης, με τα δικά του samples ο καθένας. Οι 9 εθελοντές του test
δεν συμμετέχουν ποτέ στην εκπαίδευση. Σε αυτούς αξιολογείται κάθε μοντέλο.

Το ίδιο δίκτυο (561-256-128-6) εκπαιδεύεται με τρεις τρόπους. Κεντρικά, σε όλα τα δεδομένα μαζί.
Ομοσπονδιακά, με FedAvg (McMahan κ.ά. 2017), όπου σε κάθε γύρο οι clients εκπαιδεύουν τοπικά και ο server
παίρνει τον σταθμισμένο μέσο όρο των μοντέλων τους. Ατομικά, με κάθε client να εκπαιδεύει μόνος του.

Για την αποτίμηση, κάθε γύρος της ομοσπονδιακής εκπαίδευσης ορίζει ένα συνεργατικό παίγνιο με παίκτες
τους 21 clients.

- **Η αξία v(S)** ενός συνασπισμού S είναι το accuracy στο test του μοντέλου που θα έβγαινε στον γύρο, αν
  είχαν συνεισφέρει μόνο οι clients του S. Το μοντέλο αυτό δεν εκπαιδεύεται από την αρχή. Φτιάχνεται από τα
  τοπικά μοντέλα του γύρου με τη στάθμιση του FedAvg (Liu κ.ά. 2022), οπότε κάθε v(S) κοστίζει μία
  αξιολόγηση. Το v(∅) είναι το accuracy του μοντέλου στην αρχή του γύρου και το v(N) στο τέλος του.
- **Η τιμή Shapley** φ_i του client i στον γύρο (Shapley 1953) είναι

  φ_i = Σ_{S ⊆ N∖{i}} |S|! (n − |S| − 1)! / n! × [v(S ∪ {i}) − v(S)]

  όπου n = 21 το πλήθος των clients και S κάθε συνασπισμός χωρίς τον i. Η αγκύλη είναι το πόσο αλλάζει το
  accuracy όταν ο i προστίθεται στους S. Το κλάσμα είναι η πιθανότητα, αν οι clients μπαίνουν με τυχαία
  σειρά, να έχουν μπει πριν από τον i ακριβώς οι S. Άρα το φ_i είναι η μέση αλλαγή που φέρνει ο i, πάνω σε
  όλες τις σειρές εισόδου. Οι 21 τιμές αθροίζουν στο v(N) − v(∅). Για παράδειγμα, αν σε έναν γύρο το
  accuracy ανέβει από 0,90 σε 0,92, τα 21 φ του γύρου αθροίζουν 0,02. Ένα αρνητικό φ σημαίνει ότι ο client
  κατά μέσο όρο μείωσε το accuracy όταν προστέθηκε.
- **Η συνεισφορά** κάθε client είναι το άθροισμα των φ_i του σε όλους τους γύρους, μέσος όρος σε 20 seeds.

Ο ακριβής υπολογισμός θέλει 2^21, δηλαδή περίπου 2,1 εκατομμύρια, αξιολογήσεις ανά γύρο. Γι' αυτό το φ
εκτιμάται με δύο εκτιμητές στο ίδιο παίγνιο, τον KernelSHAP της βιβλιοθήκης shap (Lundberg και Lee 2017)
με 2.090 συνασπισμούς ανά γύρο και τον GTG-Shapley (Liu κ.ά. 2022) με έως 100 τυχαίες σειρές εισόδου.
Γύροι όπου το accuracy άλλαξε λιγότερο από 0,001, περίπου 3 από τα 2.947 samples του test, παραλείπονται.
Η συμφωνία των δύο μετριέται κυρίως με τον Kendall τ της κατάταξης των clients. Το τ παίρνει τιμές από −1
έως 1 και είναι ο αριθμός των ζευγών clients που οι δύο εκτιμητές κατατάσσουν με την ίδια σειρά, μείον
όσα κατατάσσουν ανάποδα, προς το σύνολο των ζευγών. Το 1 σημαίνει ίδια κατάταξη. Σε τρεις γύρους του
seed 0 οι εκτιμητές συγκρίνονται και με το ακριβές Shapley, από όλους τους συνασπισμούς.

Τα υπόλοιπα scripts της αποτίμησης χωρίζουν το φ ανά δραστηριότητα, ανά άτομο του test και ανά φάση της
εκπαίδευσης. Υπολογίζουν Shapley και στην κεντρική εκπαίδευση, με νέα εκπαίδευση για κάθε συνασπισμό
(Ghorbani και Zou 2019). Επανεκπαιδεύουν χωρίς τους clients με αρνητική τιμή, για να φανεί αν το μοντέλο
βελτιώνεται. Τέλος, μετατρέπουν τις τιμές σε μερίδια ανταμοιβής.

## Οργάνωση

Ένας φάκελος ανά ρόλο, όπως στο αποθετήριο αναφοράς Simple_FL_Example (Pavlidis 2023).

| φάκελος | περιεχόμενο |
|---|---|
| `implementation/dataset/` | ανάγνωση των ωμών αρχείων, έλεγχοι, διερευνητική ανάλυση, εξαγωγή και φόρτωση των επεξεργασμένων δεδομένων |
| `implementation/common/` | το δίκτυο, ο βρόχος εκπαίδευσης και οι μετρικές, κοινά για όλες τις τοπολογίες |
| `implementation/centralized/` | κεντρική εκπαίδευση και κλασικά μοντέλα σύγκρισης |
| `implementation/federated/` | ομοσπονδιακή εκπαίδευση (`fl/`: client, server, FedAvg) |
| `implementation/local_only/` | ατομική εκπαίδευση |
| `implementation/contribution/` | αποτίμηση της συνεισφοράς (`sv/`: παίγνιο, εκτιμητές, αξιολόγηση, πολιτικές μεριδίων) |
| `data/har_processed/` | τα επεξεργασμένα δεδομένα, αντίγραφο του `../datasets/har_processed/` (βλ. Δεδομένα) |
| `results/` | εδώ γράφονται τα αποτελέσματα, με έναν υποφάκελο ανά ρόλο |

## Εγκατάσταση

```bash
pip install -r requirements.txt
```

Χρησιμοποίησα Python 3.13. Όλες οι εντολές τρέχουν από τον φάκελο που περιέχει το `implementation/`.

## Δεδομένα

Τα δεδομένα είναι στον φάκελο `datasets/har_processed/` των παραδοτέων. Τα πειράματα τα διαβάζουν από το
`data/har_processed/`, οπότε πριν από το πρώτο τρέξιμο αντιγράφονται εκεί:

```bash
mkdir -p data && cp -R ../datasets/har_processed data/
```

Ο φάκελος `data/har_processed/` περιέχει τότε τα δεδομένα όπως τα διαβάζουν τα πειράματα.

- `X_train.csv`, `X_test.csv`: τα 561 χαρακτηριστικά, με z-score από τον μέσο και την τυπική απόκλιση
  μόνο του train
- `y_train.csv`, `y_test.csv`: η δραστηριότητα (1-6)
- `client_ids_subject.csv`: ο client (0-20) κάθε sample του train
- `subject_ids_test.csv`: ο εθελοντής κάθε sample του test, με τον αριθμό του στο UCI
- `feature_names.csv`: τα ονόματα των χαρακτηριστικών
- `PROVENANCE.json`: το sha256 των ωμών αρχείων και κάθε αρχείου εδώ

Πριν διαβάσει οτιδήποτε, κάθε script ελέγχει το sha256 των αρχείων απέναντι στο `PROVENANCE.json`.

Για το στάδιο των δεδομένων (έλεγχοι στα ωμά αρχεία, διερευνητική ανάλυση, εξαγωγή) χρειάζεται και το
ωμό dataset. Κατεβαίνει από το UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Το
`UCI HAR Dataset.zip` που περιέχει αποσυμπιέζεται στο `data/`, ώστε να υπάρχει ο φάκελος
`data/UCI HAR Dataset/` με τους υποφακέλους `train/` και `test/`. Η εξαγωγή ξαναγράφει τα αρχεία του
`data/har_processed/` ίδια byte προς byte.

## Αναπαραγωγή

Οι εντολές εκτελούνται με αυτή τη σειρά, από τον φάκελο που περιέχει το `implementation/`, αφού κάποιες
αναλύσεις διαβάζουν αποτελέσματα προηγούμενων. Τα αποτελέσματα γράφονται στο `results/`.

```bash
export OMP_NUM_THREADS=1

# δεδομένα (χρειάζεται το data/UCI HAR Dataset)
python -m implementation.dataset.run_eda_uci

# οι τρεις τοπολογίες μάθησης
python -m implementation.centralized.centralized
python -m implementation.centralized.centralized --model_name logreg
python -m implementation.centralized.classic_baselines
python -m implementation.federated.federated
python -m implementation.federated.federated --fl_rounds 100 --epochs 1 --seeds 0 1 2 3 4 --out results/federated/sweep_fedE_r100_e1.json
python -m implementation.federated.federated --fl_rounds 50 --epochs 2 --seeds 0 1 2 3 4 --out results/federated/sweep_fedE_r50_e2.json
python -m implementation.federated.federated --fl_rounds 20 --epochs 5 --seeds 0 1 2 3 4 --out results/federated/sweep_fedE_r20_e5.json
python -m implementation.federated.federated --fl_rounds 10 --epochs 10 --seeds 0 1 2 3 4 --out results/federated/sweep_fedE_r10_e10.json
python -m implementation.local_only.local_only

# αποτίμηση στην ομοσπονδιακή εκπαίδευση
python -m implementation.contribution.contribution
python -m implementation.contribution.contribution --value_metric macro_f1
python -m implementation.contribution.contribution --estimators gtg_guided
python -m implementation.contribution.contribution_classwise
python -m implementation.contribution.contribution_classwise --value_metric per_subject --nsamples auto --gtg_max_iters 200

# ακριβές Shapley σε τρεις γύρους (ανεξάρτητοι μεταξύ τους) και ένωση
python -m implementation.contribution.convergence --rounds 1 --m1_check --out results/contribution/exact_r1.json
python -m implementation.contribution.convergence --rounds 5 --m1_check --out results/contribution/exact_r5.json
python -m implementation.contribution.convergence --rounds 15 --m1_check --out results/contribution/exact_r15.json
python -m implementation.contribution.convergence --merge results/contribution/exact_r1.json results/contribution/exact_r5.json results/contribution/exact_r15.json

# Shapley στην κεντρική εκπαίδευση (παίγνιο επανεκπαίδευσης)
python -m implementation.contribution.data_shapley --pilot
python -m implementation.contribution.data_shapley
python -m implementation.contribution.data_shapley --summarize
python -m implementation.contribution.verify_exact

# αναλύσεις πάνω στα παραπάνω (το label_tv θέλει το results/dataset/eda_uci.json)
python -m implementation.contribution.label_tv
python -m implementation.contribution.profile_clients
python -m implementation.contribution.update_direction
python -m implementation.contribution.temporal_profile
python -m implementation.contribution.classwise_groups
python -m implementation.contribution.per_subject_analysis

# επανεκπαίδευση χωρίς τους clients με αρνητική τιμή
python -m implementation.contribution.removal
python -m implementation.contribution.removal --analyze results/contribution/removal_parts/part.json

python -m implementation.contribution.reward_table
```

Οι χρόνοι είναι από τα δικά μου τρεξίματα σε έναν πυρήνα CPU (με
`OMP_NUM_THREADS=1`), εκτός αν γράφεται αλλιώς. Τα seeds είναι ανεξάρτητα, οπότε τα αργά scripts
μοιράζονται σε παράλληλες διεργασίες με `--seeds` και δικό τους `--out`. Μετά το `--merge` ενώνει τα
αρχεία χωρίς να ξανατρέξει κάτι.

| αποτέλεσμα στο `results/` | script | χρόνος | ενότητα |
|---|---|---|---|
| `dataset/raw_audit_uci.json` | `dataset/inspect_raw_uci.py` | 15 s | 3.3 |
| `dataset/eda_uci.json` | `dataset/eda_uci.py` | 10 s | 3.3, 3.4 |
| `dataset/scaling_check_uci.json` | `dataset/scaling_check.py` | 5 s | 3.3.1 |
| `centralized/har_central_ann.json` | `centralized/centralized.py` | 6 λεπτά | 4.3.1, 4.6 |
| `centralized/har_central_logreg.json` | `centralized/centralized.py --model_name logreg` | 6 λεπτά | 4.5 |
| `centralized/har_central_classic.json` | `centralized/classic_baselines.py` | 11 λεπτά | 4.5 |
| `federated/har_fed_ann.json` | `federated/federated.py` | 8 λεπτά | 4.3.2, 4.6, 4.7 |
| `federated/sweep_fedE_*.json` | `federated/federated.py` με τοπικές εποχές × γύρους = 100, 5 seeds | 10 λεπτά και τα τέσσερα | 4.6.3 |
| `local_only/har_local_ann.json` | `local_only/local_only.py` | 12 λεπτά | 4.3.3, 4.6 |
| `contribution/har_contribution_ann_20seeds.json` | `contribution/contribution.py` | 50 λεπτά ανά seed | 5.3 |
| `contribution/har_contribution_ann_20seeds_macrof1.json` | `contribution/contribution.py --value_metric macro_f1` | 45 λεπτά ανά seed | 5.4 |
| `contribution/har_contribution_guided_20seeds.json` | `contribution/contribution.py --estimators gtg_guided` | 8 λεπτά ανά seed | 5.2.4 |
| `contribution/har_contribution_exact_rounds.json` | `contribution/convergence.py` | 5 ώρες ανά γύρο | 5.2.4, 5.2.5, 5.3.2 |
| `contribution/har_contribution_classwise.json` | `contribution/contribution_classwise.py` | 55 λεπτά ανά seed | 5.7.2 |
| `contribution/har_contribution_per_subject.json` | `contribution/contribution_classwise.py --value_metric per_subject` | 40 λεπτά ανά seed | 5.7.3 |
| `contribution/har_data_shapley.json` | `contribution/data_shapley.py` | 1,8 ώρες ανά seed με 6 διεργασίες | 5.6 |
| `contribution/har_data_shapley_firstlast_20seeds.json` | `contribution/data_shapley.py --pilot` | 2 λεπτά ανά seed με 6 διεργασίες | 5.6.4 |
| `contribution/har_data_shapley_exact_check.json` | `contribution/verify_exact.py` | 5 λεπτά με 6 διεργασίες | 5.2.5 |
| `contribution/har_label_tv.json` | `contribution/label_tv.py` | 5 s | 5.5.1 |
| `contribution/har_client_profiles.json` | `contribution/profile_clients.py` | 10 s | 5.5.1, 5.5.2 |
| `contribution/har_update_direction.json` | `contribution/update_direction.py` | 20 λεπτά | 5.5.3 |
| `contribution/har_contribution_temporal.json` | `contribution/temporal_profile.py` | 5 s | 5.7.1 |
| `contribution/har_classwise_groups.json` | `contribution/classwise_groups.py` | 1 s | 5.7.2 |
| `contribution/har_contribution_per_subject_analysis.json` | `contribution/per_subject_analysis.py` | 1 s | 5.7.3 |
| `contribution/har_removal.json` | `contribution/removal.py` | 12 λεπτά ανά seed | 5.7.4, 5.7.5 |
| `contribution/har_reward_shares.json` | `contribution/reward_table.py` | 1 s | 5.8 |

Με τις εκδόσεις του `requirements.txt` και σε CPU, ο κώδικας δίνει τα ίδια νούμερα με εκείνα της εργασίας.
Το έλεγξα ξανατρέχοντας το seed 0 κάθε πειράματος και τις αναλύσεις πάνω στα αποθηκευμένα αποτελέσματα. Από τα δύο
ακριβά βήματα ξανάτρεξα ένα μέρος. Από το Shapley με επανεκπαίδευση (`data_shapley.py`) ξανάτρεξα ολόκληρο το seed 0
(2.092 εκπαιδεύσεις), το πιλοτικό του και τον έλεγχο με το ακριβές σε 8 παίκτες. Από τη σύγκριση με το ακριβές
(`convergence.py`) ξανάτρεξα το ακριβές Shapley του γύρου 1 (όλοι οι 2^21 συνασπισμοί) και όλους τους εκτιμητές στους
τρεις γύρους.

Τρεις λεπτομέρειες που επηρεάζουν την αναπαραγωγή:

- Κάθε αξιολόγηση μέσω `DataLoader` τραβά έναν αριθμό από τη γεννήτρια του torch, ακόμη και χωρίς
  ανακάτεμα. Γι' αυτό οι αξιολογήσεις γίνονται πάντα στα ίδια σημεία της εκπαίδευσης. Όπου γίνεται έλεγχος
  που δεν υπάρχει στο `federated.py`, η κατάσταση της γεννήτριας κρατιέται και επανέρχεται.
- Στο παίγνιο επανεκπαίδευσης (`data_shapley.py`, `verify_exact.py`) ο KernelSHAP τρέχει με την
  προεπιλογή της βιβλιοθήκης `l1_reg="num_features(21)"`. Στην ομοσπονδιακή εκπαίδευση τρέχει με
  `l1_reg=False`, χωρίς επιλογή παικτών.
- Στο `convergence.py` ο server δεν αξιολογεί το μοντέλο σε κάθε γύρο, όπως έτρεξε η σύγκριση με το
  ακριβές. Οι γύροι 5 και 15 εκεί δεν είναι ακριβώς οι γύροι 5 και 15 του `contribution.py`, ενώ ο γύρος 1
  είναι ίδιος.

## Πηγές

- Anguita, Ghio, Oneto, Parra, Reyes-Ortiz (2013). A Public Domain Dataset for Human Activity Recognition
  Using Smartphones. ESANN. https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf
- Reyes-Ortiz, Anguita, Ghio, Oneto, Parra (2013). Human Activity Recognition Using Smartphones. UCI
  Machine Learning Repository. https://doi.org/10.24432/C54S4K
- McMahan, Moore, Ramage, Hampson, Agüera y Arcas (2017). Communication-Efficient Learning of Deep Networks
  from Decentralized Data. AISTATS. https://arxiv.org/abs/1602.05629
- Shapley (1953). A Value for n-Person Games. Contributions to the Theory of Games II, 307-318.
  https://doi.org/10.1515/9781400881970-018
- Lundberg, Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 30.
  https://arxiv.org/abs/1705.07874
- Liu, Chen, Yu, Liu, Cui (2022). GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation
  in Federated Learning. ACM TIST 13(4). https://doi.org/10.1145/3501811
- Ghorbani, Zou (2019). Data Shapley: Equitable Valuation of Data for Machine Learning. ICML.
  https://arxiv.org/abs/1904.02868
- Tastan, Fares, Aremu, Horvath, Nandakumar (2024). Redefining Contributions: Shapley-Driven Federated
  Learning. IJCAI-24, 5009-5017. https://www.ijcai.org/proceedings/2024/0554.pdf
- Yang, Jarin, Buyukates, Avestimehr, Markopoulou (2024). Maverick-Aware Shapley Valuation for Client
  Selection in Federated Learning. https://arxiv.org/abs/2405.12590
- Karimireddy, Kale, Mohri, Reddi, Stich, Suresh (2020). SCAFFOLD: Stochastic Controlled Averaging for
  Federated Learning. ICML. https://arxiv.org/abs/1910.06378
- Jaynes (1957). Information Theory and Statistical Mechanics. Physical Review 106(4), 620-630.
  https://doi.org/10.1103/PhysRev.106.620
- Pavlidis (2023). Simple Federated Learning Example. https://github.com/nikopavl4/Simple_FL_Example
