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
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.numpy_function(fill_annotation, [mask_path], tf.uint8)
    mask.set_shape([None, None, 1])
    return img, mask
#------------------------
def fill_annotation(path):
    ann = cv2.imread(path.decode(), cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours((ann > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(ann)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask[..., None]
#----------------------
def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    return input_image, input_mask

image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)


    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

input_size=(96, 128, 3)
n_filters = 32
inputs = Input(input_size)
next_layer, skip_connection = conv_block(inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=inputs, outputs=next_layer)

output1 = [
    ['InputLayer', (None, 96, 128, 3), 0],
    ['Conv2D', (None, 96, 128, 32), 896, 'same', 'relu', 'HeNormal'],
    ['Conv2D', (None, 96, 128, 32), 9248, 'same', 'relu', 'HeNormal'],
    ['MaxPooling2D', (None, 48, 64, 32), 0, (2, 2)]
]

print('Block 1:')
for layer in summary(model1):
    print(layer)

comparator(summary(model1), output1)
inputs = Input(input_size)
next_layer, skip_connection = conv_block(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
model2 = tf.keras.Model(inputs=inputs, outputs=next_layer)
output2 = [
    ['InputLayer', (None, 96, 128, 3), 0],
    ['Conv2D', (None, 96, 128, 1024), 28672, 'same', 'relu', 'HeNormal'],
    ['Conv2D', (None, 96, 128, 1024), 9438208, 'same', 'relu', 'HeNormal'],
    ['Dropout', (None, 96, 128, 1024), 0, 0.1],
    ['MaxPooling2D', (None, 48, 64, 1024), 0, (2, 2)]
]

print('\nBlock 2:')
for layer in summary(model2):
    print(layer)

comparator(summary(model2), output2)
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(
                 n_filters,
                 3,
                 strides=2,
                 padding='same')(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)
    return conv

input_size1=(12, 16, 256)
input_size2 = (24, 32, 128)
n_filters = 32
expansive_inputs = Input(input_size1)
contractive_inputs =  Input(input_size2)
cblock1 = upsampling_block(expansive_inputs, contractive_inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=cblock1)
output1 = [
    ['InputLayer', (None, 12, 16, 256), 0],
    ['Conv2DTranspose', (None, 24, 32, 32), 73760],
    ['InputLayer', (None, 24, 32, 128), 0],
    ['Concatenate', (None, 24, 32, 160), 0],
    ['Conv2D', (None, 24, 32, 32), 46112, 'same', 'relu', 'HeNormal'],
    ['Conv2D', (None, 24, 32, 32), 9248, 'same', 'relu', 'HeNormal']
]


print('Block 1:')
for layer in summary(model1):
    print(layer)
comparator(summary(model1), output1)
def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=2):
    inputs = Input(input_size)

    # Contracting Path
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Expanding Path
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters*8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding='same', activation='softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model

img_height = 96
img_width = 128
num_channels = 3

unet = unet_model((img_height, img_width, num_channels))
unet.summary()
unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
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
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

plt.plot(model_history.history["accuracy"])
def show_predictions(dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
             create_mask(unet.predict(sample_image[tf.newaxis, ...]))])

show_predictions(train_dataset, 6)