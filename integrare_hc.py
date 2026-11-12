import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from biometrie import incarca, extrage_contururi, selecteaza_contur, potriveste_elipsa
from metrice import dice_score
from metrice import DiceMetric
from predictIndependent import predict_single_image
from calibrare import citeste_calibrare
from biometrie import calculeaza_hc

# 1 incarc modelul antrenat de colega
model = tf.keras.models.load_model("unet_fetal_head.keras",
                                   custom_objects={"DiceMetric": DiceMetric})

# 2 aleg o imagine
nume = "000_HC.png"
cale = "Data/HC18/training_set/" + nume

# 3 U-Net prezice masca
_, masca_prezisa, dim_originala = predict_single_image(model, cale)

# 4 calculez HC din masca prezisa
pixel_size, hc_ref = citeste_calibrare()[nume]
rezultat = calculeaza_hc(masca_prezisa, pixel_size, dim_originala)

print("Dimensiune masca prezisa:", masca_prezisa.shape)
print("HC medic:", hc_ref, "mm")
print("HC din masca U-Net:", rezultat["hc_mm"], "mm")
print("Motiv:", rezultat["motiv"])
# 5 masca medicului pentru aceeasi imagine
img, masca_ref = incarca(nume)
rezultat_ref = calculeaza_hc(masca_ref, pixel_size, img.shape[:2])
print("HC din masca medicului:", rezultat_ref["hc_mm"], "mm")

# 6 cat de bine se suprapun cele doua masti
print("Dice:", dice_score(masca_ref[..., 0], masca_prezisa))

# 7 desenez cele doua elipse peste ecografie
def elipsa(masca):
    contur = selecteaza_contur(extrage_contururi(masca), masca.shape[0] * masca.shape[1])
    return potriveste_elipsa(contur)

desen = img.copy()
cv2.ellipse(desen, elipsa(masca_ref), (0, 255, 0), 3)
cv2.ellipse(desen, elipsa(masca_prezisa), (0, 0, 255), 2)

fig, axe = plt.subplots(1, 2, figsize=(14, 5))
axe[0].imshow(masca_prezisa, cmap="gray")
axe[0].set_title("Masca prezisa de U-Net")
axe[1].imshow(desen[..., ::-1])
axe[1].set_title("Verde medic   Rosu U-Net")
for ax in axe:
    ax.axis("off")
plt.tight_layout()
plt.show()