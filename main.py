import numpy as np
from termcolor import colored

from tensorflow.keras.layers import Input

from config import (SEED, image_path, mask_path, EPOCHS,
                    VAL_SUBSPLITS, BUFFER_SIZE, BATCH_SIZE,
                    set_seeds, save_config, split_train_val)
set_seeds()

from unet import build_model
unet = build_model()
unet.summary()

import tensorflow as tf
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras import Model
import cv2
# from google.colab import drive


# drive.mount('/content/drive')

image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png') and not f.endswith('_Annotation.png')]
mask_list = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('_Annotation.png')]

image_list.sort()
mask_list.sort()

assert len(image_list) == len(mask_list), "Număr diferit de imagini și masti"
for img_p, msk_p in zip(image_list, mask_list):
    assert os.path.basename(img_p).replace('.png', '') == \
           os.path.basename(msk_p).replace('_Annotation.png', ''), \
           f"Nepotrivire: {img_p} / {msk_p}"
print(f"OK: {len(image_list)} perechi verificate")

save_config(image_list, mask_list)
train_images, train_masks, val_images, val_masks = split_train_val(image_list, mask_list)
N = 2
img = imageio.imread(image_list[N])
mask = imageio.imread(mask_list[N])

fig, arr = plt.subplots(1, 2, figsize=(14, 10))
arr[0].imshow(img)
arr[0].set_title('Image')
arr[0].axis('off')

if mask.ndim == 3:
    arr[1].imshow(mask[:, :, 0], cmap='gray')
else:
    arr[1].imshow(mask, cmap='gray')
arr[1].set_title('Segmentation')
arr[1].axis('off')
plt.tight_layout()
plt.show()
image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
    print(path)

image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)

dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

for image, mask in dataset.take(1):
    print(image)
    print(mask)

from preprocessing import image_ds, processed_image_ds


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
for image, mask in image_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])
#new

for image, mask in processed_image_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])

from data_loader import process_path
from preprocessing import preprocess

train_ds = tf.data.Dataset.from_tensor_slices(
    (tf.constant(train_images), tf.constant(train_masks))).map(process_path).map(preprocess)
val_ds = tf.data.Dataset.from_tensor_slices(
    (tf.constant(val_images), tf.constant(val_masks))).map(process_path).map(preprocess)

processed_image_ds.batch(BATCH_SIZE)
train_dataset = train_ds.cache().shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE)
val_dataset = val_ds.batch(BATCH_SIZE)
print(processed_image_ds.element_spec)
model_history = unet.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

from metrice import evaluate_dataset
val_dice, val_iou = evaluate_dataset(unet, val_dataset)
from evaluate import create_mask, plot_history, show_predictions

plot_history(model_history)
show_predictions(unet, display, train_dataset, 6)