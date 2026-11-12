import os
import tensorflow as tf

from config import (SEED, image_path, mask_path, EPOCHS, BUFFER_SIZE, BATCH_SIZE,
                    set_seeds, save_config, split_train_val_test)
from data_loader import process_path
from preprocessing import preprocess
from unet import build_model
from metrice import DiceMetric, evaluate_dataset
from evaluate import (display, show_predictions, save_training_log,
                      plot_training_history)

MODEL_PATH = "unet_fetal_head.keras"
PATIENCE = 8

set_seeds()

# 1 lista perechilor imagine masca si verificarea lor
image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)
                     if f.endswith(".png") and not f.endswith("_Annotation.png")])
mask_list = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path)
                    if f.endswith("_Annotation.png")])

assert len(image_list) == len(mask_list), "Numar diferit de imagini si masti"
for img_p, msk_p in zip(image_list, mask_list):
    assert os.path.basename(img_p).replace(".png", "") == \
           os.path.basename(msk_p).replace("_Annotation.png", ""), \
           f"Nepotrivire: {img_p} / {msk_p}"
print(f"OK: {len(image_list)} perechi verificate")

save_config(image_list, mask_list)

# 2 impartirea train validation test dupa pacient
train_images, train_masks, val_images, val_masks, test_images, test_masks = \
    split_train_val_test(image_list, mask_list)


def make_dataset(images, masks):
    return tf.data.Dataset.from_tensor_slices(
        (tf.constant(images), tf.constant(masks))).map(process_path).map(preprocess)


train_dataset = make_dataset(train_images, train_masks).cache() \
    .shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE)
val_dataset = make_dataset(val_images, val_masks).cache().batch(BATCH_SIZE)
test_dataset = make_dataset(test_images, test_masks).batch(BATCH_SIZE)

# 3 modelul nou de la zero
unet = build_model()
unet.summary()

# 4 antrenare cu oprire automata si salvarea celui mai bun model
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_dice", mode="max",
                                     patience=PATIENCE, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_dice",
                                       mode="max", save_best_only=True),
]

model_history = unet.fit(train_dataset, epochs=EPOCHS,
                         validation_data=val_dataset, callbacks=callbacks)
save_training_log(model_history)

# 5 reincarc cel mai bun model salvat si il evaluez
best = tf.keras.models.load_model(MODEL_PATH, custom_objects={"DiceMetric": DiceMetric})
print(f"\nCel mai bun model a fost salvat in: {MODEL_PATH}")

val_dice, val_iou = evaluate_dataset(best, val_dataset)
test_dice, test_iou = evaluate_dataset(best, test_dataset, path="test_metrics.txt")

# 6 graficele la final ca sa nu opreasca antrenarea
plot_training_history(model_history)
show_predictions(best, display, test_dataset, 3)