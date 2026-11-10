import os
import cv2
import matplotlib.pyplot as plt
from biometrie import (incarca, extrage_contururi, selecteaza_contur,
                       potriveste_elipsa, FOLDER_FIGURI)

IMAGINI = ["000_HC.png", "022_HC.png", "100_HC.png",
           "300_HC.png", "637_HC.png", "805_HC.png"]


def test_elipsa(nume_imagini):
    fig, axe = plt.subplots(2, 3, figsize=(15, 8))
    for ax, nume in zip(axe.flat, nume_imagini):
        img, masca = incarca(nume)
        arie_imagine = masca.shape[0] * masca.shape[1]
        #contur ales si elipsa potrivita pe el
        capul = selecteaza_contur(extrage_contururi(masca), arie_imagine)
        elipsa = potriveste_elipsa(capul)
        #despachetez elipsa ca sa afisez valorile
        (cx, cy), (d1, d2), unghi = elipsa
        print(f"{nume}  centru = ({cx:.0f}, {cy:.0f})  "
              f"diametre = {d1:.1f} x {d2:.1f} px  unghi = {unghi:.1f}")
        #verde gros pentru contur si rosu subtire pentru elipsa
        cv2.drawContours(img, [capul], -1, (0, 255, 0), 4)
        cv2.ellipse(img, elipsa, (0, 0, 255), 2)
        ax.imshow(img[..., ::-1])
        ax.set_title(f"{nume}\naxe {d1:.0f} x {d2:.0f} px, unghi {unghi:.0f}",fontsize=10)
        ax.axis("off")
    fig.suptitle("Verde contur extras   Rosu elipsa potrivita", fontsize=13)
    plt.tight_layout()
    os.makedirs(FOLDER_FIGURI, exist_ok=True)
    plt.savefig(os.path.join(FOLDER_FIGURI, "task5_elipse.png"), dpi=150)
    plt.show()

def test_detaliu(nume):
    #zoom pe o portiune din margine ca sa se vada scara de pixeli
    img, masca = incarca(nume)
    arie_imagine = masca.shape[0] * masca.shape[1]
    capul = selecteaza_contur(extrage_contururi(masca), arie_imagine)
    elipsa = potriveste_elipsa(capul)
    cv2.drawContours(img, [capul], -1, (0, 255, 0), 1)
    cv2.ellipse(img, elipsa, (0, 0, 255), 1)
    #aleg o fereastra de 60 x 60 pixeli in jurul celui mai de sus punct al conturului
    x, y = capul[capul[:, 0, 1].argmin()][0]
    decupaj = img[max(y - 30, 0):y + 30, max(x - 30, 0):x + 30]
    #figura noua ca sa nu desenez peste gridul de mai sus
    plt.figure(figsize=(6, 6))
    plt.imshow(decupaj[..., ::-1], interpolation="nearest")
    plt.title(f"{nume} zoom pe margine\nverde scara de pixeli   rosu elipsa neteda")
    plt.axis("off")
    plt.savefig(os.path.join(FOLDER_FIGURI, "task5_zoom.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    test_elipsa(IMAGINI)
    test_detaliu("300_HC.png")