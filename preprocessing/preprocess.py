import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def apply_clahe(image):
    """Applies CLAHE to enhance contrast in MRI images."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def normalize_image(image):
    """Normalizes the image to the range [0, 1]."""
    return image / 255.0

def augment_images(image_array):
    """Applies data augmentation to images."""
    data_gen_args = dict(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True)
    datagen = ImageDataGenerator(**data_gen_args)
    it = datagen.flow(image_array, batch_size=1)
    return it.next()[0]  # Return the augmented image
