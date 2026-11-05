import tensorflow as tf
import cv2
import numpy as np

def fill_annotation(path):
    ann = cv2.imread(path.decode(), cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours((ann > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(ann)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask[..., None]
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.numpy_function(fill_annotation, [mask_path], tf.uint8)
    mask.set_shape([None, None, 1])
    return img, mask
