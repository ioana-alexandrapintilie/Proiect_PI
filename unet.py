import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Dropout, Conv2DTranspose, concatenate
)
from __main__ import summary, comparator

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
