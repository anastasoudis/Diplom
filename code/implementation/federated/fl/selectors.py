# Ποιοι clients συμμετέχουν σε κάθε γύρο.


class AllSelector:
    """Όλοι οι clients σε κάθε γύρο. Αυτό χρησιμοποιούν όλα τα πειράματα της εργασίας."""

    def __repr__(self):
        return "AllSelector()"

    def sample_clients(self, client_list):
        return list(client_list)


class ExcludeSelector:
    """Όλοι εκτός από κάποιους συγκεκριμένους clients, σε κάθε γύρο.

    Για το πείραμα αφαίρεσης (removal.py). Οι αποκλεισμένοι δεν εκπαιδεύονται και
    δεν μπαίνουν στη συνάθροιση. Ένας αριθμός client που δεν υπάρχει είναι σφάλμα,
    γιατί αλλιώς το τρέξιμο θα γινόταν σιωπηλά με όλους.
    """

    def __init__(self, exclude=()):
        self.exclude = frozenset(int(i) for i in exclude)
        if not self.exclude:
            raise ValueError("ExcludeSelector χωρίς αποκλεισμούς")

    def __repr__(self):
        return f"ExcludeSelector(exclude={sorted(self.exclude)})"

    def sample_clients(self, client_list):
        unknown = self.exclude - {cl.id for cl in client_list}
        if unknown:
            raise ValueError(f"άγνωστοι clients {sorted(unknown)}")
        kept = [cl for cl in client_list if cl.id not in self.exclude]
        if not kept:
            raise ValueError("αποκλείστηκαν όλοι οι clients")
        return kept


def get_selector(name: str, exclude=None):
    if name == "all":
        if exclude:
            raise ValueError("το --exclude ισχύει μόνο με --selector exclude")
        return AllSelector()
    if name == "exclude":
        return ExcludeSelector(exclude or ())
    raise ValueError(f"άγνωστος selector {name!r}")
