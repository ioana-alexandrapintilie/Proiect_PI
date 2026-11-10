import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_loader import fill_annotation
from calibrare import citeste_calibrare, FOLDER

FOLDER_FIGURI = "figuri"

def extrage_contururi(masca):
    # fill_annotation intoarce forma H W 1 iar findContours vrea H W
    if masca.ndim == 3:
        masca = masca[..., 0]

    contururi, _ = cv2.findContours(
        masca.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    return contururi

def incarca(nume):
    # intoarce ecografia si masca plina pentru un nume de imagine
    img = cv2.imread(os.path.join(FOLDER, nume))
    cale_masca = os.path.join(FOLDER, nume.replace(".png", "_Annotation.png"))
    masca = fill_annotation(cale_masca.encode())
    return img, masca


def selecteaza_contur(contururi, arie_imagine, prag=0.01):
    # daca nu exista niciun contur nu exista cap
    if len(contururi) == 0:
        return None

    # caut conturul cu aria cea mai mare
    arie_max = cv2.contourArea(contururi[0])
    contur_selectat = contururi[0]
    for c in contururi:
        arie = cv2.contourArea(c)
        if arie > arie_max:
            arie_max = arie
            contur_selectat = c

    # daca si cel mai mare contur e prea mic nu il consider cap
    if arie_max < prag * arie_imagine:
        return None
    return contur_selectat


def potriveste_elipsa(contur):
    # fara contur nu am ce potrivi
    if contur is None:
        return None

    # fitEllipse are nevoie de cel putin 5 puncte
    if len(contur) < 5:
        return None

    # elipsa are forma centru diametre unghi
    return cv2.fitEllipse(contur)


# TESTE VIZUALE

def test_contur(nume_imagini):
    cal = citeste_calibrare()
    fig, axe = plt.subplots(2, 3, figsize=(15, 8))

    for ax, nume in zip(axe.flat, nume_imagini):
        img, masca = incarca(nume)
        arie_imagine = masca.shape[0] * masca.shape[1]
        contururi = extrage_contururi(masca)
        capul = selecteaza_contur(contururi, arie_imagine)
        cv2.drawContours(img, [capul], -1, (0, 255, 0), 2)

        # lungimea conturului in pixeli comparata cu valoarea din CSV
        pixel_size, hc_ref = cal[nume]
        lungime_px = cv2.arcLength(capul, True)
        asteptat_px = hc_ref / pixel_size
        diferenta = (lungime_px - asteptat_px) / asteptat_px * 100

        ax.imshow(img[..., ::-1])
        ax.set_title(f"{nume} | contururi: {len(contururi)}\n"
                     f"{lungime_px:.0f} px vs. {asteptat_px:.0f} px "
                     f"({diferenta:+.1f}%)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(FOLDER_FIGURI, exist_ok=True)
    plt.savefig(os.path.join(FOLDER_FIGURI, "task3_contururi.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    test_contur(["000_HC.png", "022_HC.png", "100_HC.png",
                 "300_HC.png", "637_HC.png", "805_HC.png"])