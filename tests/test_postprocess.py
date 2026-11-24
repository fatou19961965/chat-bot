from app.postprocess import apply_lexicon, code_switch_ratio, is_question, restore_punctuation


def test_question_wolof():
    assert is_question("naka nga def")
    assert restore_punctuation("naka nga def") == "Naka nga def?"
    assert restore_punctuation("dama bëgg lekk") == "Dama bëgg lekk."
    assert restore_punctuation("") == ""


def test_lexique():
    txt, n = apply_lexicon("jerejef bu baax", {"jerejef": "jërëjëf"})
    assert txt == "jërëjëf bu baax" and n == 1


def test_alternance_codique():
    assert code_switch_ratio("dama bëgg ubbi compte ci banque bi") > 0
    assert code_switch_ratio("") == 0.0
