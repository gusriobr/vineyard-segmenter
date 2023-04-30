import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


def augment_image(image, mask):
    image = tf.cast(image, dtype=tf.float16)
    mask = tf.cast(mask, dtype=tf.uint8)
    return image, mask
    # # Convertir la máscara a tf.float32
    # mask = tf.cast(mask, tf.float32)
    # # Aplicar aumentos aleatorios a la imagen y la máscara
    # image = tf.image.random_brightness(image, 0.2)
    # image = tf.image.random_contrast(image, 0.8, 1.2)
    # mask = tf.image.random_flip_left_right(mask)
    # # Concatenar la imagen y la máscara a lo largo del último eje
    # augmented = tf.concat([image, mask], axis=-1)
    # # Devolver la imagen aumentada y la máscara original
    # return augmented[:, :, :3], augmented[:, :, 3:]


def augment_image2(image, mask):
    # Aplica transformaciones de aumento de datos a la imagen y la máscara
    image_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    seed = tf.random.uniform([], dtype=tf.int32, maxval=2 ** 31 - 1)
    # Concatena la imagen y la máscara para que se transformen juntas
    mask = tf.cast(mask, tf.float32)
    combined = tf.concat([image, mask], axis=-1)
    # Convierte la imagen y la máscara concatenadas a un array de NumPy
    combined = combined.numpy()
    # Aplica las transformaciones de aumento de datos a la imagen y la máscara
    combined = image_datagen.random_transform(combined, seed=seed)
    # Separa la imagen y la máscara
    image, mask = combined[..., :3], combined[..., 3:]
    # Normaliza la imagen y la máscara
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask
