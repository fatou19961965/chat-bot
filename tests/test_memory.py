from app.memory import TranslationMemory


def _tm(tmp_path):
    cur = tmp_path / "cur.tsv"
    cur.write_text("# commentaire\nnaka nga def\tComment vas-tu ?\njërëjëf\tMerci.\n", encoding="utf-8")
    return TranslationMemory(cur, tmp_path / "user.tsv")


def test_exacte_insensible_casse_ponctuation_diacritiques(tmp_path):
    tm = _tm(tmp_path)
    assert tm.lookup("Naka nga def?").francais == "Comment vas-tu ?"
    m = tm.lookup("jerejef")
    assert m.francais == "Merci." and m.exacte


def test_approchee_faute_de_frappe(tmp_path):
    m = _tm(tmp_path).lookup("naka nua def")
    assert m is not None and not m.exacte and m.francais == "Comment vas-tu ?"


def test_pas_de_faux_positif(tmp_path):
    assert _tm(tmp_path).lookup("dama bëgg dem ca marse ba") is None


def test_ajout_utilisateur_persiste(tmp_path):
    tm = _tm(tmp_path)
    assert tm.add("dama xiif", "J'ai faim.")
    assert not tm.add("naka nga def", "autre")          # liste validée protégée
    tm2 = TranslationMemory(tmp_path / "cur.tsv", tmp_path / "user.tsv")
    assert tm2.lookup("Dama xiif.").francais == "J'ai faim."
