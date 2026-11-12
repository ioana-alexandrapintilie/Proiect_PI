import numpy as np
from termcolor import colored

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

from unet import conv_block, upsampling_block


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


def run_tests():
    #conv_block, Block 1
    input_size = (96, 128, 3)
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

    #conv_block, Block 2
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

    #upsampling_block
    input_size1 = (12, 16, 256)
    input_size2 = (24, 32, 128)
    n_filters = 32
    expansive_inputs = Input(input_size1)
    contractive_inputs = Input(input_size2)
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


if __name__ == "__main__":
    run_tests()