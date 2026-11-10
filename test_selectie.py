import cv2
import numpy as np
from biometrie import incarca, extrage_contururi, selecteaza_contur

#incarc ecografia si masca plina
img, masca = incarca("300_HC.png")
arie_imagine = masca.shape[0] * masca.shape[1]

#caz 0 masca curata cu un singur contur
contururi_curate = extrage_contururi(masca)
capul = selecteaza_contur(contururi_curate, arie_imagine)
print("Masca curata, aria aleasa:", cv2.contourArea(capul))

#adaug zgomot artificial peste o copie a mastii
masca_zgomot = masca[..., 0].copy()
cv2.circle(masca_zgomot, (60, 60), 15, 1, -1)
cv2.circle(masca_zgomot, (740, 480), 25, 1, -1)
cv2.circle(masca_zgomot, (700, 80), 8, 1, -1)

#afisez toate contururile gasite in masca cu zgomot
contururi = extrage_contururi(masca_zgomot)
print("\nNumar de contururi:", len(contururi))
for i, c in enumerate(contururi):
    print(f"Conturul {i}: aria = {cv2.contourArea(c):.0f} px, puncte = {len(c)}")

#caz 1 cap si zgomot trebuie ales capul
capul = selecteaza_contur(contururi, arie_imagine)
print("\nCu zgomot, aria aleasa:", cv2.contourArea(capul))

#caz 2 doar zgomot fara cap trebuie None
doar_zgomot = np.zeros((540, 800), dtype=np.uint8)
cv2.circle(doar_zgomot, (60, 60), 15, 1, -1)
cv2.circle(doar_zgomot, (740, 480), 25, 1, -1)
print("Doar zgomot:", selecteaza_contur(extrage_contururi(doar_zgomot), arie_imagine))

#caz 3 masca complet goala trebuie None
goala = np.zeros((540, 800), dtype=np.uint8)
print("Masca goala:", selecteaza_contur(extrage_contururi(goala), arie_imagine))