import tensorflow as tf
import matplotlib.pyplot as plt


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def plot_history(model_history):
    plt.plot(model_history.history["accuracy"])
    plt.show()


def show_predictions(model, display, dataset=None, num=1,
                     sample_image=None, sample_mask=None):
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])