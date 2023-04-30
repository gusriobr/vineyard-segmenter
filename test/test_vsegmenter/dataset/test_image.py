import unittest
import numpy as np
import tensorflow as tf

from data.image import augment_image


class TestAugmentImage(unittest.TestCase):
    def test_augment_image(self):
        image_shape = (256, 256, 3)
        mask_shape = (256, 256, 1)
        image = tf.random.uniform(image_shape, dtype=tf.float32)
        mask = tf.random.uniform(mask_shape, dtype=tf.float32)
        # Convierte la imagen y la máscara a tensores de TensorFlow
        image = tf.convert_to_tensor(image)
        mask = tf.convert_to_tensor(mask)
        # Aplica la función de aumento de datos a la imagen y la máscara
        augmented_image, augmented_mask = augment_image(image, mask)
        # Comprueba que las dimensiones de la imagen y la máscara son correctas
        self.assertEqual(augmented_image.shape, (256, 256, 3))
        self.assertEqual(augmented_mask.shape, (256, 256, 1))
        # Comprueba que la imagen y la máscara tienen valores en el rango correcto
        self.assertGreaterEqual(tf.reduce_min(augmented_image), 0.0)
        self.assertLessEqual(tf.reduce_max(augmented_image), 1.0)
        self.assertGreaterEqual(tf.reduce_min(augmented_mask), 0.0)
        self.assertLessEqual(tf.reduce_max(augmented_mask), 1.0)
        # Comprueba que la imagen y la máscara son diferentes de las originales
        self.assertFalse(tf.reduce_all(tf.equal(augmented_image, image)))
        self.assertFalse(tf.reduce_all(tf.equal(augmented_mask, mask)))
