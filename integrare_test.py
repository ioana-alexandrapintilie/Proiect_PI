import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from config import image_path, mask_path, split_train_val_test
from metrice import DiceMetric, dice_score
from predictIndependent import predict_single_image
from calibrare import citeste_calibrare
from biometrie import (incarca, calculeaza_hc, extrage_contururi,
                       selecteaza_contur, potriveste_elipsa, FOLDER_FIGURI)

MODEL = "unet_fetal_head.keras"
FOLDER_REZULTATE = "rezultate"


def imagini_test():
    # aceeasi impartire ca la antrenare citita din split_info_3.txt
    image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)
                         if f.endswith(".png") and not f.endswith("_Annotation.png")])
    mask_list = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path)
                        if f.endswith("_Annotation.png")])
    _, _, _, _, test_images, _ = split_train_val_test(image_list, mask_list, path="split_info_3.txt")
    return [os.path.basename(p) for p in test_images]


def elipsa_px(masca):
    # elipsa in pixeli doar pentru desenat
    capul = selecteaza_contur(extrage_contururi(masca), masca.shape[0] * masca.shape[1])
    return potriveste_elipsa(capul)


def main():
    print("Se incarca modelul salvat")
    model = tf.keras.models.load_model(MODEL, custom_objects={"DiceMetric": DiceMetric})

    cal = citeste_calibrare()
    nume_test=imagini_test()
    nume_test = nume_test[:20]
    print(f"Imagini de test: {len(nume_test)}")

    randuri = []
    exemple = []
    for nume in nume_test:
        print("procesez", nume, flush=True)
        # predictie U-Net cu masca readusa la dimensiunea originala
        _, masca_pred, dim = predict_single_image(model, os.path.join(image_path, nume))

        # masca de referinta si calibrarea
        img, masca_ref = incarca(nume)
        pixel_size, hc_ref = cal[nume]

        # HC din masca prezisa si din masca de referinta
        rez_pred = calculeaza_hc(masca_pred, pixel_size, dim)
        rez_ref = calculeaza_hc(masca_ref, pixel_size, img.shape[:2])

        randuri.append({"imagine": nume,
                        "dice": dice_score(masca_ref[..., 0], masca_pred),
                        "hc_ref": hc_ref,
                        "hc_din_masca_ref": rez_ref["hc_mm"],
                        "hc_din_masca_prezisa": rez_pred["hc_mm"],
                        "motiv": rez_pred["motiv"]})

        if len(exemple) < 6:
            exemple.append((nume, img, masca_ref, masca_pred, hc_ref, rez_pred["hc_mm"]))

    df = pd.DataFrame(randuri)
    df["eroare_prezisa_mm"] = df["hc_din_masca_prezisa"] - df["hc_ref"]
    df["eroare_ref_mm"] = df["hc_din_masca_ref"] - df["hc_ref"]

    os.makedirs(FOLDER_REZULTATE, exist_ok=True)
    df.round(3).to_csv(os.path.join(FOLDER_REZULTATE, "integrare_test.csv"), index=False)

    valide = df.dropna(subset=["hc_din_masca_prezisa", "hc_din_masca_ref"])
    print(f"\nImagini evaluate: {len(df)}, cu HC valid din ambele masti: {len(valide)}")
    print(f"Dice mediu (rezolutie originala): {df['dice'].mean():.4f}")
    print(f"MAE din masca de referinta: {valide['eroare_ref_mm'].abs().mean():.2f} mm")
    print(f"MAE din masca prezisa:      {valide['eroare_prezisa_mm'].abs().mean():.2f} mm")
    print(f"Eroare procentuala medie din masca prezisa: "
          f"{(valide['eroare_prezisa_mm'].abs() / valide['hc_ref'] * 100).mean():.2f} %")

    print("\nImagini fara HC din masca prezisa:")
    print(df[df["hc_din_masca_prezisa"].isna()][["imagine", "dice", "motiv"]].to_string(index=False))

    print("\nCele mai mari 5 erori din masca prezisa:")
    print(valide.reindex(valide["eroare_prezisa_mm"].abs().sort_values(ascending=False).index)
          .head(5)[["imagine", "dice", "hc_ref", "hc_din_masca_prezisa", "eroare_prezisa_mm"]]
          .round(2).to_string(index=False))

    # figura verde elipsa din masca de referinta rosu elipsa din masca prezisa
    fig, axe = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (nume, img, masca_ref, masca_pred, hc_ref, hc_pred) in zip(axe.flat, exemple):
        desen = img.copy()
        e_ref = elipsa_px(masca_ref)
        e_pred = elipsa_px(masca_pred)
        if e_ref is not None:
            cv2.ellipse(desen, e_ref, (0, 255, 0), 3)
        if e_pred is not None:
            cv2.ellipse(desen, e_pred, (0, 0, 255), 2)
        text_pred = f"{hc_pred:.1f}" if hc_pred is not None else "invalid"
        ax.imshow(desen[..., ::-1])
        ax.set_title(f"{nume}\nHC medic {hc_ref:.1f} mm | HC U-Net {text_pred} mm", fontsize=11)
        ax.axis("off")
    fig.suptitle("Verde elipsa din masca de referinta   Rosu elipsa din masca prezisa de U-Net", fontsize=13)
    plt.tight_layout()
    os.makedirs(FOLDER_FIGURI, exist_ok=True)
    plt.savefig(os.path.join(FOLDER_FIGURI, "integrare_exemple.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()