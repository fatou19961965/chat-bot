from eval.agreement import weighted_kappa


def test_kappa():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0
