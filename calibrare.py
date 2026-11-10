import os
from collections import Counter
import cv2
import pandas as pd


FOLDER = "Data/HC18/training_set"
CSV = "Data/HC18/training_set_pixel_size_and_HC.csv"


def citeste_calibrare(cale_csv=CSV):
    df = pd.read_csv(cale_csv)
    calibrare = {}
    for _, rand in df.iterrows():
        calibrare[rand["filename"]] = (rand["pixel size(mm)"], rand["head circumference (mm)"])
    return calibrare


def verifica_potrivire(calibrare):
    adnotari = [f for f in os.listdir(FOLDER) if f.endswith("_Annotation.png")]
    din_masti = {f.replace("_Annotation.png", ".png") for f in adnotari}
    din_csv = set(calibrare.keys())
    print("Adnotari in folder", len(din_masti))
    print("Randuri in csv", len(din_csv))
    print("au masca, dar lipsesc din csv: ", din_masti - din_csv)
    print("sunt in csv, dar nu au masca", din_csv - din_masti)


def statistici(cale_csv=CSV):
    df = pd.read_csv(cale_csv)
    # imi afisez pentru fiecare coloana
    # media, deviatia std, min, cuantile si max
    # si vf daca exista val lipsa pe fiecare coloana
    print("\nvalori lipsa pe coloane:\n", df.isna().sum())
    print("\nPixel size:\n", df['pixel size(mm)'].describe())
    print("\nHC referinta:\n", df['head circumference (mm)'].describe())


def dim_img():
    imagini = [f for f in os.listdir(FOLDER)
               if f.endswith(".png") and not f.endswith("_Annotation.png")]
    dimensiuni = Counter()
    for nume in imagini:
        img = cv2.imread(os.path.join(FOLDER, nume), cv2.IMREAD_GRAYSCALE)
        dimensiuni[img.shape] += 1
    print("\nDimensiuni imagini (inaltime, latime) nr. imagini")
    for dim, nr in dimensiuni.items():
        print(" ", dim, " ", nr)


if __name__ == "__main__":
    cal = citeste_calibrare()
    verifica_potrivire(cal)
    statistici()
    dim_img()