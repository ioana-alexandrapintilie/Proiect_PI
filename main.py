import numpy as np
from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
def comparator(learner, instructor):
    if len(learner) != len(instructor):
        raise AssertionError(f"The number of layers in the proposed model does not agree with the expected model: expected {len(instructor)}, got {len(learner)}.")
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"),
                  "\n\n does not match the input value: \n\n",
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test")
    print(colored("All tests passed!", "green"))
def summary(model):
    result = []
    for layer in model.layers:
        # Safely get the output shape
        try:
            output_shape = layer.output.shape
        except AttributeError:
            output_shape = getattr(layer, 'batch_input_shape', 'unknown')

        # Safely get parameters
        try:
            params = layer.count_params()
        except:
            params = 0

        descriptors = [layer.__class__.__name__, output_shape, params]

        if isinstance(layer, Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if isinstance(layer, MaxPooling2D):
            descriptors.append(layer.pool_size)
        if isinstance(layer, Dropout):
            descriptors.append(layer.rate)

        result.append(descriptors)
    return result

from unet import unet_model, unet

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

image_path = r"Data/HC18/training_set"
mask_path = r"Data/HC18/training_set"
image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png') and not f.endswith('_Annotation.png')]
mask_list = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('_Annotation.png')]

image_list.sort()
mask_list.sort()
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


EPOCHS = 5
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
processed_image_ds.batch(BATCH_SIZE)
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)
model_history = unet.fit(train_dataset, epochs=EPOCHS)

from evaluate import create_mask, show_predictions
