import hashlib
import random

import numpy as np
import tensorflow as tf
import os

SEED = 42

image_path = r"Data/HC18/training_set"
mask_path = r"Data/HC18/training_set"

EPOCHS = 5
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
VAL_FRACTION = 0.2
input_size = (96, 128, 3)


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def save_config(image_list, mask_list, path="experiment_config.txt"):
    pairs = list(zip(image_list, mask_list))
    pairs_hash = hashlib.md5(str(pairs).encode()).hexdigest()

    content = (
        f"seed={SEED}\n"
        f"epochs={EPOCHS}\n"
        f"batch_size={BATCH_SIZE}\n"
        f"buffer_size={BUFFER_SIZE}\n"
        f"val_subsplits={VAL_SUBSPLITS}\n"
        f"input_size={input_size}\n"
        f"pairs_hash={pairs_hash}\n"
    )
    for img, msk in pairs:
        content += f"{img} {msk}\n"

    if os.path.exists(path):
        with open(path) as f:
            saved = f.read()
        assert saved == content, (
            "Configuratia sau perechile difera de cele salvate in "
            f"{path}"
        )
        print("aceleasi perechi si aceeasi configuratie")
    else:
        with open(path, "w") as f:
            f.write(content)
        print(f"configuratia a fost salvata in {path}")

    print("Hash perechi:", pairs_hash)


def patient_id(path):
    return os.path.basename(path).split("_")[0]


def split_train_val(image_list, mask_list, path="split_info.txt"):
    ids = sorted({patient_id(p) for p in image_list})
    random.Random(SEED).shuffle(ids)
    n_val = int(len(ids) * VAL_FRACTION)
    val_ids = set(ids[:n_val])

    train_images = [p for p in image_list if patient_id(p) not in val_ids]
    train_masks = [p for p in mask_list if patient_id(p) not in val_ids]
    val_images = [p for p in image_list if patient_id(p) in val_ids]
    val_masks = [p for p in mask_list if patient_id(p) in val_ids]

    # verificări
    assert not ({patient_id(p) for p in train_images} &
                {patient_id(p) for p in val_images}), "Pacienti in ambele seturi"
    assert not (set(train_images) & set(val_images)), "Imagini in ambele seturi"
    assert len(train_images) + len(val_images) == len(image_list), "Imagini pierdute"
    assert len(train_images) == len(train_masks) and len(val_images) == len(val_masks)
    for i, m in zip(train_images + val_images, train_masks + val_masks):
        assert os.path.basename(i).replace('.png', '') == \
               os.path.basename(m).replace('_Annotation.png', ''), f"Nepotrivire: {i} / {m}"

    # documentare
    content = (
        f"seed={SEED}\n"
        f"val_fraction={VAL_FRACTION}\n"
        f"pacienti_train={len(ids) - len(val_ids)}\n"
        f"pacienti_val={len(val_ids)}\n"
        f"imagini_train={len(train_images)}\n"
        f"imagini_val={len(val_images)}\n"
    )
    for i in train_images:
        content += f"TRAIN {i}\n"
    for i in val_images:
        content += f"VAL {i}\n"

    if os.path.exists(path):
        with open(path) as f:
            saved = f.read()
        assert saved == content, f"Impartirea difera de cea salvata în {path}"
        print("impartire reproductibila ok")
    else:
        with open(path, "w") as f:
            f.write(content)
        print(f"impartirea a fost salvata în {path}")

    print(f"Pacienti: train={len(ids) - len(val_ids)}, val={len(val_ids)}")
    print(f"Imagini:  train={len(train_images)}, val={len(val_images)}")
    return train_images, train_masks, val_images, val_masks