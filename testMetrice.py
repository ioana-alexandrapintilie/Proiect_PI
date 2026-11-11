import numpy as np
from metrice import dice_score, iou_score


def check(name, got, expected):
    assert np.isclose(got, expected), f"{name}: obtinut {got}, asteptat {expected}"
    print(f"OK  {name}: {got:.4f}")


def run_tests():
    full = np.array([[1, 1], [1, 1]])
    empty = np.zeros((2, 2))

    #Suprapunere completa
    check("complet - Dice", dice_score(full, full), 1.0)
    check("complet - IoU", iou_score(full, full), 1.0)

    #Suprapunere nula
    a = np.array([[1, 1], [0, 0]])
    b = np.array([[0, 0], [1, 1]])
    check("nul - Dice", dice_score(a, b), 0.0)
    check("nul - IoU", iou_score(a, b), 0.0)

    #Suprapunere partiala: intersectie 1, |A|=2, |B|=1
    c = np.array([[1, 1], [0, 0]])
    d = np.array([[1, 0], [0, 0]])
    check("partial - Dice", dice_score(c, d), 2 * 1 / 3)
    check("partial - IoU", iou_score(c, d), 1 / 2)

    #Masca goala
    check("ambele goale - Dice", dice_score(empty, empty), 1.0)
    check("ambele goale - IoU", iou_score(empty, empty), 1.0)
    check("referinta goala, predictie nenula - Dice", dice_score(empty, full), 0.0)
    check("referinta goala, predictie nenula - IoU", iou_score(empty, full), 0.0)
    check("referinta nenula, predictie goala - Dice", dice_score(full, empty), 0.0)
    check("referinta nenula, predictie goala - IoU", iou_score(full, empty), 0.0)

    print("All tests passed!")


if __name__ == "__main__":
    run_tests()