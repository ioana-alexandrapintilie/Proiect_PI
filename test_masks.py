import cv2
import numpy as np
import os
from data_loader import fill_annotation

def main():
    folder_path = "Data/HC18/training_set"
    #citesc si sortez fisierele din folder
    files = sorted(os.listdir(folder_path))
    #automatizez verificarea generarii mastilor binare
    verificate = 0
    problematice = []

    for x in files:
        if x.endswith("_Annotation.png"):

            #construiesc numele ecografiei corespunzatoare
            image_name = x.replace("_Annotation.png", ".png")
            image_path = os.path.join(folder_path, image_name)
            #citesc ecografia originala
            image = cv2.imread(image_path)
            #construiesc calea completa catre adnotare
            file_path = os.path.join(folder_path, x)
            #verific daca ecografia poate fi citita
            if image is None:
                print("ECOGRAFIA NU POATE FI CITITA:", image_name)
                problematice.append(x)
                verificate += 1
                continue
            #apelez functia existenta din data_loader.py
            try:
                annotation = fill_annotation(file_path.encode())
            except Exception as e:
                print("EROARE LA CITIREA ADNOTARII:", x, e)
                problematice.append(x)
                verificate += 1
                continue
            verificate += 1
            #verific valorile mastii
            valori_corecte = np.all(np.isin(annotation, [0, 1]))
            #compar dimensiunile mastii cu cele ale ecografiei
            dimensiuni_corecte = (
                annotation.shape[:2] == image.shape[:2]
                and annotation.ndim == 3
                and annotation.shape[2] == 1
            )
            #verific daca masca nu este complet neagra
            masca_nevida = np.count_nonzero(annotation) > 0
            #afisez motivul fiecarei probleme
            if not valori_corecte:
                print("VALORI INVALIDE:", x, np.unique(annotation))
            if not dimensiuni_corecte:
                print("DIMENSIUNI DIFERITE:", x)
                print("  Ecografie:", image.shape)
                print("  Masca:", annotation.shape)
            if not masca_nevida:
                print("MASCA GOALA:", x)
            #adaug o singura data fisierul problematic
            if not (valori_corecte and dimensiuni_corecte and masca_nevida):
                problematice.append(x)

    #raport final
    print("\nAdnotari verificate:", verificate)
    print("Adnotari problematice:", len(problematice))
    print("Fisiere problematice:", problematice)


if __name__ == '__main__':
    main()