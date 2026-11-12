import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from config import image_path, mask_path, split_train_val_test
from metrice import DiceMetric, dice_score
from predictIndependent import predict_single_image
from calibrare import citeste_calibrare
from biometrie import (incarca, calculeaza_hc, extrage_contururi,
                       selecteaza_contur, potriveste_elipsa)

MODEL="unet_fetal_head.keras"
NUMAR_IMAGINI = 3


def imagini_test():
    # imaginile de test din aceeasi impartire ca la antrenare
    image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)
                         if f.endswith(".png") and not f.endswith("_Annotation.png")])
    mask_list = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path)
                        if f.endswith("_Annotation.png")])
    _, _, _, _, test_images, _ = split_train_val_test(image_list, mask_list, path="split_info_3.txt")
    return [os.path.basename(p) for p in test_images]


def elipsa(masca):
    # elipsa in pixeli doar pentru desen
    contur = selecteaza_contur(extrage_contururi(masca), masca.shape[0] * masca.shape[1])
    return potriveste_elipsa(contur)


def text_hc(valoare):
    return "invalid" if valoare is None else f"{valoare:.1f} mm"


def demo_imagine(model, nume, pixel_size, hc_ref):
    # pasul 1 U-Net prezice masca capului
    _, masca_prezisa, dim_originala = predict_single_image(model, os.path.join(image_path, nume))

    # pasul 2 calculul HC din masca prezisa
    rezultat = calculeaza_hc(masca_prezisa, pixel_size, dim_originala)

    # pentru comparatie masca medicului
    img, masca_ref = incarca(nume)
    rezultat_ref = calculeaza_hc(masca_ref, pixel_size, img.shape[:2])
    dice = dice_score(masca_ref[..., 0], masca_prezisa)

    print(f"\n{nume}")
    print(f"  Dice segmentare:        {dice:.3f}")
    print(f"  HC medic (CSV):         {hc_ref:.1f} mm")
    print(f"  HC din masca medicului: {text_hc(rezultat_ref['hc_mm'])}")
    print(f"  HC din masca U-Net:     {text_hc(rezultat['hc_mm'])}")
    if rezultat["motiv"] is not None:
        print(f"  Motiv: {rezultat['motiv']}")

    # desenez elipsele verde medic rosu U-Net
    desen = img.copy()
    e_ref = elipsa(masca_ref)
    e_pred = elipsa(masca_prezisa)
    if e_ref is not None:
        cv2.ellipse(desen, e_ref, (0, 255, 0), 3)
    if e_pred is not None:
        cv2.ellipse(desen, e_pred, (0, 0, 255), 2)

    fig, axe = plt.subplots(1, 3, figsize=(18, 5))
    axe[0].imshow(img[..., ::-1])
    axe[0].set_title("1 Ecografia")
    axe[1].imshow(masca_prezisa, cmap="gray")
    axe[1].set_title(f"2 Masca prezisa de U-Net (Dice {dice:.2f})")
    axe[2].imshow(desen[..., ::-1])
    axe[2].set_title(f"3 Verde medic {hc_ref:.1f} mm   Rosu U-Net {text_hc(rezultat['hc_mm'])}")
    for ax in axe:
        ax.axis("off")
    fig.suptitle(f"{nume}   pixel size {pixel_size:.4f} mm", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    print("Se incarca modelul U-Net antrenat")
    model = tf.keras.models.load_model(MODEL, custom_objects={"DiceMetric": DiceMetric})

    cal = citeste_calibrare()
    nume_test = imagini_test()[:NUMAR_IMAGINI]
    print(f"Demo pe {len(nume_test)} imagini din setul de test, nevazute la antrenare")

    for nume in nume_test:
        pixel_size, hc_ref = cal[nume]
        demo_imagine(model, nume, pixel_size, hc_ref)


if __name__ == "__main__":
    main()