from app.metrics import cer, chrf, chrf_pp, corpus_wer, normalize_text, wer, wer_stats


def test_normalisation_ponctuation_casse_balises():
    assert normalize_text("Salaamaalekum, NAKA nga def ?") == "salaamaalekum naka nga def"
    assert normalize_text("Dama bëgg <fr>compte</fr> [bruit] euh bi.") == "dama bëgg compte bi"


def test_diacritiques_conserves_par_defaut():
    assert normalize_text("jërëjëf") == "jërëjëf"
    assert normalize_text("jërëjëf ŋ", strip_diacritics=True) == "jerejef n"


def test_apostrophe_interne():
    assert normalize_text("j'ai « dit »") == "j'ai dit"


def test_wer_valeurs_connues():
    assert wer("a b c d", "a b c d") == 0.0
    assert wer("a b c d", "a x c d") == 0.25
    st = wer_stats("a b c d", "a c d e")
    assert (st.deletions, st.insertions) == (1, 1)
    assert wer("", "") == 0.0
    assert wer("", "a") == 1.0


def test_wer_corpus_pondere_par_longueur():
    # 1 erreur sur 1 mot + 0 sur 9 mots = 10 %, pas 50 %
    assert abs(corpus_wer(["a", "b c d e f g h i j"], ["x", "b c d e f g h i j"]) - 0.1) < 1e-9


def test_cer():
    assert abs(cer("abcd", "abxd") - 0.25) < 1e-9


def test_chrf_bornes():
    assert chrf("bonjour le monde", "bonjour le monde") == 100.0
    assert chrf("", "bonjour") == 0.0
    s = chrf_pp("bonjour le monde", "bonjour à tous")
    assert 0 < s < 100
    assert chrf("bonjour le monde", "bonjour le mondes") > chrf("bonsoir", "bonjour le mondes")
