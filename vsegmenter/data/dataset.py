import os
import pickle

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

import cfg


def create_sample(image_path, mask_path, image_size=None):
    """
    Creates sample from image and mask file paths
    :param image_path:
    :param mask_path:
    :return:
    """
    # load image and mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image_size is not None:
        image = image.resize((image_size, image_size))
        mask = mask.resize((image_size, image_size))
    # load into numpy array
    image = np.array(image)
    mask = np.array(mask)

    # normalize image and mask
    image = image / 255.0
    mask = mask / 255.0
    # transform to binary
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    # unet dataset needs a different channel for each category. Channel 0 = background 1 = foreground
    unet_mask = np.zeros(mask.shape + (2,))
    unet_mask[..., 0] = (mask == 0)  # background category
    unet_mask[..., 1] = (mask == 1)
    # truncate datatype to reduce memory footage
    unet_mask = unet_mask.astype(np.uint8)
    image = image.astype(np.float16)

    return {"image": image, "mask": unet_mask}


def create_dataset(train_images, test_images):
    info = {"num_train": len(train_images[0]), "num_test": len(test_images[0])}
    return {"train": train_images, "test": test_images, "info": info}


def save_dataset(dts, dest_file):
    info = {"num_train": len(dts["train"][0]), "num_test": len(dts["test"][0])}
    dts["info"] = info  # recalculate
    with open(dest_file, "wb") as f:
        pickle.dump(dts, f)


def load_dataset(dataset_file):
    with open(dataset_file, "rb") as f:
        return pickle.load(f)


def create_segmentation_dataset(image_dir, image_size):
    """
    Crea un dataset para entrenar un modelo UNET a partir de imágenes y máscaras.

    Args:
        image_dir: La ruta al directorio que contiene las imágenes.
        mask_dir: La ruta al directorio que contiene las máscaras.
        image_size: El tamaño deseado de las imágenes (ancho, alto).
        batch_size: El tamaño del batch para entrenamiento.

    Returns:
        Un objeto tf.data.Dataset que contiene las imágenes y máscaras.
    """

    # Obtener la lista de nombres de archivo de las imágenes y máscaras
    image_names = sorted([f for f in os.listdir(image_dir) if "data" in f])
    mask_names = sorted([f for f in os.listdir(image_dir) if "mask" in f])

    assert len(image_names) == len(mask_names), "the list of imagen aren't same size"

    image_paths = [os.path.join(image_dir, name) for name in image_names]
    mask_paths = [os.path.join(image_dir, name) for name in mask_names]

    images, masks = create_array_dataset(image_paths, mask_paths, image_size)

    train_idx, test_idx = split_indexes(image_paths, train_ratio=0.8)

    train_dataset = np.stack([images[i] for i in train_idx], axis=0), np.stack([masks[i] for i in train_idx], axis=0)
    test_dataset = np.stack([images[i] for i in test_idx], axis=0), np.stack([masks[i] for i in test_idx], axis=0)

    return train_dataset, test_dataset


def create_tf_dataset(image_paths, mask_paths, batch_size):
    images = []
    masks = []
    data = []
    for image_filename, mask_filename in zip(image_paths, mask_paths):
        sample = create_sample(image_filename, mask_filename)
        images.append(sample["image"])
        masks.append(sample["mask"])
    # create dataset from image and mask lists
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size)
    return dataset


def create_array_dataset(image_paths, mask_paths, image_size):
    images = []
    masks = []
    for image_filename, mask_filename in zip(image_paths, mask_paths):
        sample = create_sample(image_filename, mask_filename, image_size=image_size)
        images.append(sample["image"])
        masks.append(sample["mask"])
    # create dataset from image and mask lists
    return images, masks


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def split_indexes(dataset, train_ratio=0.6, test_ratio=0.2, validation_split=False, shuffle=True, seed=None):
    # Get the total number of examples in the dataset
    num_examples = len(dataset)

    # Get the indices of the examples
    indices = list(range(num_examples))

    # Optionally shuffle the indices
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    # Calculate the sizes of each split
    train_size = int(train_ratio * num_examples)
    test_size = num_examples - train_size if validation_split is False else int(test_ratio * num_examples)
    val_size = 0 if validation_split is False else num_examples - train_size - test_size

    # Divide the indices into three groups for the splits
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    if val_size > 0:
        val_indices = indices[train_size + test_size:train_size + test_size + val_size]

    if validation_split:
        return train_indices, test_indices, val_indices
    else:
        return train_indices, test_indices


if __name__ == '__main__':
    image_dir = "/tmp/samples"
    image_size = 128
    batch_size = 64

    make_dataset = True
    dts_file = cfg.resource("dataset/dataset.pickle")
    if make_dataset:
        train_samples, test_samples = create_segmentation_dataset(image_dir, image_size)
        dataset = create_dataset(train_samples, test_samples)
        save_dataset(dataset, dts_file)
    else:
        dataset = load_dataset(dts_file)
        train_samples = dataset["train"]
        test_samples = dataset["test"]

    dataset = load_dataset(dts_file)
