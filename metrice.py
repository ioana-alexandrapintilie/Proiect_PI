import numpy as np


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