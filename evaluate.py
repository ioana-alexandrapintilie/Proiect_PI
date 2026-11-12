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

from config import SEED, EPOCHS, BATCH_SIZE


def save_training_log(model_history, path="training_log.txt"):
    h = model_history.history
    n_epochs_run = len(h["loss"])
    with open(path, "w") as f:
        f.write(f"seed={SEED}\n")
        f.write(f"epochs_configurate={EPOCHS}\n")
        f.write(f"epochs_rulate={n_epochs_run}\n")
        f.write(f"batch_size={BATCH_SIZE}\n")
        f.write("epoca,loss,val_loss,dice,val_dice,accuracy,val_accuracy\n")
        for e in range(n_epochs_run):
            f.write(f"{e + 1},{h['loss'][e]:.4f},{h['val_loss'][e]:.4f},"
                    f"{h['dice'][e]:.4f},{h['val_dice'][e]:.4f},"
                    f"{h['accuracy'][e]:.4f},{h['val_accuracy'][e]:.4f}\n")
    print(f"istoricul antrenarii a fost salvat in {path}")


def plot_training_history(model_history, path="training_history.png"):
    h = model_history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, key, title in zip(ax, ["loss", "dice", "accuracy"],
                             ["Loss", "Dice", "Accuracy"]):
        a.plot(epochs, h[key], label="train")
        a.plot(epochs, h["val_" + key], label="validation")
        a.set_title(title)
        a.set_xlabel("Epoca")
        a.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()