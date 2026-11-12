import numpy as np
import tensorflow as tf

def dice_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    total = y_true.sum() + y_pred.sum()
    if total == 0:
        return 1.0
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2.0 * intersection / total


def iou_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(y_true, y_pred).sum()
    return intersection / union


def evaluate_dataset(model, dataset, path="validation_metrics.txt"):
    dices, ious = [], []
    for image, mask in dataset:
        pred = np.argmax(model.predict(image, verbose=0), axis=-1)   # (B, H, W)
        true = mask.numpy()[..., 0]                                  # (B, H, W)
        for t, p in zip(true, pred):
            dices.append(dice_score(t, p))
            ious.append(iou_score(t, p))

    mean_dice = float(np.mean(dices))
    mean_iou = float(np.mean(ious))

    with open(path, "w") as f:
        f.write(f"imagini_evaluate={len(dices)}\n")
        f.write(f"dice={mean_dice:.4f}\n")
        f.write(f"iou={mean_iou:.4f}\n")

    print(f"Validare ({len(dices)} imagini): Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")
    return mean_dice, mean_iou

class DiceMetric(tf.keras.metrics.Metric):
    """Dice per imagine, mediat pe imagini (clasa 1 = capul fetal)."""

    def __init__(self, name="dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.cast(y_true[..., 0], tf.float32)                          # (B, H, W)
        pred = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)         # (B, H, W)
        inter = tf.reduce_sum(true * pred, axis=[1, 2])
        total = tf.reduce_sum(true, axis=[1, 2]) + tf.reduce_sum(pred, axis=[1, 2])
        dice = tf.where(total == 0, 1.0, 2.0 * inter / tf.maximum(total, 1.0))
        self.dice_sum.assign_add(tf.reduce_sum(dice))
        self.count.assign_add(tf.cast(tf.shape(dice)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.dice_sum, self.count)

    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)