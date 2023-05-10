import imgaug.augmenters as iaa
import numpy as np

from vsegmenter import cfg
import tensorflow as tf
cfg.configLog()


def green_variation(image, random_state, parents, hooks):
    img = np.multiply(image, [1.0, np.random.uniform(0.95, 1.05), 1.0])
    return np.clip(img, 0, 255).astype(np.uint8)

seq = iaa.Sequential([
    iaa.Fliplr(0.5, random_state=1),
    iaa.Rot90((1, 3), random_state=2),
    iaa.Sometimes(
        0.5, iaa.Affine(scale={"x": (1, 1.3), "y": (1, 1.3)}, random_state=4), random_state=5),
    iaa.Sometimes(
        0.5, iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.LinearContrast((0.85, 1.2)),
        ])
    ),
], random_state=9)

seq_mask = iaa.Sequential([
    iaa.Fliplr(0.5, random_state=1),
    iaa.Rot90((1, 3), random_state=2),
    iaa.Sometimes(
        0.5, iaa.Affine(scale={"x": (1, 1.3), "y": (1, 1.3)}, random_state=4), random_state=5),
], random_state=9)


# Definir una función para aplicar las transformaciones a cada muestra
def apply_imgaug_augmentation(image, mask):
    image = seq.augment_image(image)
    mask = seq_mask.augment_image(mask)
    return image, mask

def imgaug_transform_segmentation(image, mask):
    # Convertir los tensores a matrices de NumPy
    image_np = image.numpy()
    mask_np = mask.numpy()

    # Aplicar las transformaciones
    image_aug = seq.augment_image(image_np)
    mask_aug = seq_mask.augment_image(mask_np)

    # Convertir las matrices de NumPy de nuevo a tensores
    return tf.convert_to_tensor(image_aug, dtype=tf.float32), tf.convert_to_tensor(mask_aug, dtype=tf.float32)

@tf.function
def tensorflow_imgaug_transform(image, mask):
    [image, mask] = tf.py_function(imgaug_transform_segmentation, [image, mask], [tf.float32, tf.float32])
    return image, mask

def augment_image(image, mask):
    image = tf.cast(image, dtype=tf.float16)
    mask = tf.cast(mask, dtype=tf.uint8)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    return image, mask