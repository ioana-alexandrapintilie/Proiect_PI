import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from config import image_path, mask_path, split_train_val_test
from metrice import DiceMetric


def predict_single_image(model, img_path, model_input_size=(96, 128)):

    orig_img = tf.io.read_file(img_path)
    orig_img = tf.image.decode_png(orig_img, channels=3)
    orig_img = tf.image.convert_image_dtype(orig_img, tf.float32)

    orig_h, orig_w = orig_img.shape[0], orig_img.shape[1]

    img_resized = tf.image.resize(orig_img, model_input_size, method='nearest')
    img_batch = tf.expand_dims(img_resized, 0)

    pred = model.predict(img_batch, verbose=0)

    pred_mask = tf.argmax(pred, axis=-1)[0]

    pred_mask_3d = tf.expand_dims(pred_mask, -1)
    restored_mask = tf.image.resize(pred_mask_3d, (orig_h, orig_w), method='nearest')

    restored_mask = tf.squeeze(restored_mask).numpy()

    return orig_img.numpy(), restored_mask, (orig_h, orig_w)


def main():
    print("Se încarca modelul salvat")
    model_save_path = "unet_fetal_head.keras"
    loaded_unet = tf.keras.models.load_model(
        model_save_path,
        custom_objects={'DiceMetric': DiceMetric}
    )

    print("\nExtragem setul de validare")
    image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if
                         f.endswith('.png') and not f.endswith('_Annotation.png')])
    mask_list = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('_Annotation.png')])

    _, _, val_images, _, _, _ = split_train_val_test(image_list, mask_list, path="split_info_3.txt")

    test_images = val_images[:5]

    print(f"\nExecutam inferența pe 5 imagini...\n")

    fig, axs = plt.subplots(5, 2, figsize=(10, 20))

    for i, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)


        orig_img, restored_mask, original_size = predict_single_image(loaded_unet, img_path)


        print(f" Imagine {i + 1}: {img_name}")
        print(f"Dimensiune originala imagine : {original_size}")
        print(f"Dimensiune mască restaurata  : {restored_mask.shape}")
        print(f"Valori unice in masca        : {np.unique(restored_mask)}")  # Va fi [0 1] sau [0]

        # Plotare
        axs[i, 0].imshow(orig_img)
        axs[i, 0].set_title(f"Input ({original_size[0]}x{original_size[1]})")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(restored_mask, cmap='gray')
        axs[i, 1].set_title(f"Masca binara prezisa ({restored_mask.shape[0]}x{restored_mask.shape[1]})")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()