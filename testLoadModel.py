import tensorflow as tf
import os
import matplotlib.pyplot as plt

from metrice import DiceMetric
from evaluate import display, create_mask
from preprocessing import preprocess
from data_loader import process_path
from config import image_path, mask_path


def main():
    model_save_path = "unet_fetal_head.keras"

    print("Se incarca modelul salvat...")
    # pasam dictionarul custom_objects
    loaded_unet = tf.keras.models.load_model(
        model_save_path,
        custom_objects={'DiceMetric': DiceMetric}
    )
    print("Model incarcat cu succes! Arhitectura:")
    loaded_unet.summary()

    print("\n pregatire imagine pentru test...")
    image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if
                         f.endswith('.png') and not f.endswith('_Annotation.png')])
    mask_list = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('_Annotation.png')])

    img_path_tf = tf.constant(image_list[0])
    mask_path_tf = tf.constant(mask_list[0])

    raw_img, raw_mask = process_path(img_path_tf, mask_path_tf)
    img_prep, mask_prep = preprocess(raw_img, raw_mask)

    img_batch = tf.expand_dims(img_prep, 0)

    print(" Se realizeaza predictia (fara antrenare)...")
    pred_mask = loaded_unet.predict(img_batch)

    print("Afisare rezultate...")

    display([img_prep, mask_prep, create_mask(pred_mask)])
    print("Testare completa. Modelul functioneaza perfect din salvare!")


if __name__ == "__main__":
    main()