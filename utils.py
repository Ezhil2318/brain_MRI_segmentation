import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(path):
    images = []
    masks = []
    for img_name in os.listdir(os.path.join(path, 'images')):
        img = load_img(os.path.join(path, 'images', img_name), target_size=(256, 256), color_mode='grayscale')
        mask = load_img(os.path.join(path, 'masks', img_name), target_size=(256, 256), color_mode='grayscale')

        images.append(img_to_array(img) / 255.0)
        masks.append(img_to_array(mask) / 255.0)

    return np.array(images), np.array(masks)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
