import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from PIL import Image

import cfg
import geo.spatialite as spt
import image.pnoa as pnoa
from data.extractions import update_extraction_info
from data.sampling import run_extraction
from geo.vectors import buffer_by_distance
from image.raster import clip_raster_with_polygon
from utils.file import remake_folder, filesize_in_MB

cfg.configLog()


def get_tf_dataset(version):
    dataset_folder = cfg.dataset(f'{version}')
    x_train, y_train, x_test, y_test = Dataset.load_from_folder(dataset_folder)
    return create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)


class Dataset:
    """
    Dataset folder contains:
    - extractions: raster files used to extract the samples cutting out the sample features from the rasters of this folder
    - samples: extracted samples including image and mask
    dataset_{img_size}.pickle
    samples_{img_size}.json

    Creation process:
     0. Preparation: create dataset folder with a child folder "extractions" that contains the raster files that will be
     used to cut out samples and create rgb images and masks.
     1. Create dataset object: dts = Dataset(dataset_folder)
     1. dts.extract_samples(feature_file) -->
         - Firts burns the features from file into raster files listed in /extractions and creates /masks files with 0
         for background and 1 for features.
         - Second, samples the mask files extracting random rectangles assuring a minimum representation of each category
         (by default 30%) and stores each sample as rgb imagen and mask in /samples folder
     2. save_samples() --> creates numpy array for samples and masks and stores them in a pickle file

    Load Dataset:
     Dataset.load_dataset(folder): reads numpy dataset file and returns x_train, y_train, x_test, y_test pairs

    With x_train, y_train, x_test, y_test you can craete a tf dataset with:

    `create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)`
    """

    def __init__(self, dts_folder, img_size=128, train_ratio=0.8):
        self.img_size = img_size
        self.dts_folder = dts_folder
        self.dataset_file = f'{dts_folder}/dataset_{img_size}.pickle'
        self.samples_index = f'{dts_folder}/samples.json'
        self.extraction_folder = f'{dts_folder}/extractions'
        self.samples_folder = f'{dts_folder}/samples'
        self.masks_folder = f'{dts_folder}/masks'
        self.train_ratio = train_ratio
        self.info = {}

    def extract_rasters(self, samples_file):
        # for each extraction polygon, find its raster and give a unique name to the extraction .tiff file
        pnoa_index = cfg.pnoa("pnoa_index.sqlite")
        logging.info("Updating raster info in 'extractions' layer")
        update_extraction_info(samples_file, pnoa_index)

        # for each extraction polygon, cut-out the pnoa tile and store the tiff in the /extractions folder
        # def row_mapper:
        #     return {"pnoa_file": }
        extr_polys = spt.list_features(samples_file, "extractions", "geometry")
        csr = spt.get_srid(samples_file, "extractions", "geometry")
        for poly in extr_polys:
            raster_file = pnoa.pnoa_storage_path(poly["pnoa_tile"], cfg.PNOA_BASE_FOLDER)
            output_file = os.path.join(self.extraction_folder, poly["filename"])
            logging.info(f"Clipping PNOA tile {poly['pnoa_tile']} into file {output_file}")
            extraction_rect = poly["geometry"]
            # create 10m buffer
            # extraction_rect = buffer_by_distance(extraction_rect, csr, 5)
            clip_raster_with_polygon(extraction_rect, csr, raster_file, output_file)
        logging.info("Extraction tiff successfully created!")

    def sample_images(self, samples_file, extraction_rasters=None, sample_size=256, remake_folders=False):
        """
        Extracts images and mas
        :param samples_file: spatialite file containing feature samples
        :param extraction_rasters:
        :return:
        """
        if not os.path.exists(self.extraction_folder):
            raise ValueError(
                f"""{self.extraction_folder} not found. Create the folder {self.extraction_folder} with 
                .tiff raster files to extract sample images.""")

        self.__prepare_folders()
        if not extraction_rasters:
            # list all extractions from dataset folder
            extraction_rasters = self.get_extration_files()
            if not extraction_rasters:
                raise ValueError(f"Extraction folder {self.extraction_folder} is empty")
            # set number of samples to extract for each raster
            NUM_SAMPLES_PER_RASTER = {"0": 75, "1": 25, "mixed": 300}
            extraction_rasters = [
                [self.get_extraction_id_for_raster(file), NUM_SAMPLES_PER_RASTER, file] for file in extraction_rasters
            ]
        else:
            # check extraction files exists and add extraction id from file
            trf_extraction = []
            not_found = []
            for num_samples, filepath in extraction_rasters:
                if not os.path.exists(filepath):
                    not_found.append(filepath)
                else:
                    trf_extraction.append(
                        [self.get_extraction_id_for_raster(filepath), num_samples, filepath]
                    )
            if not_found:
                raise ValueError(f"These raster files don't exists: {not_found}")
            extraction_rasters = trf_extraction

        run_extraction(extraction_rasters, samples_file, self.dts_folder, (sample_size, sample_size),
                       remake_folders=remake_folders)

    def get_extration_files(self):
        return self._list_raster_files(self.extraction_folder)

    def _list_raster_files(self, folder):
        """
        List raster files from folder and return absolute path list
        :return:
        """
        return [os.path.join(folder, filename) for filename in
                os.listdir(folder) if "tif" in os.path.splitext(filename)[1]]

    def __prepare_folders(self):
        output_mask_folder = self.masks_folder
        samples_folder = self.samples_folder
        # re-create folders
        remake_folder(output_mask_folder)
        remake_folder(samples_folder)

    def save_samples(self):
        train_samples, test_samples, sample_dict = create_sample_arrays(self.samples_folder, self.img_size,
                                                                        train_ratio=self.train_ratio)
        logging.info(f"Dataset numpy array created")
        dts_dict = self._create_dataset_dict(train_samples, test_samples)
        self._save_dataset(dts_dict)
        logging.info(f"Dataset stored in file {self.dataset_file} ")
        self._save_sample_index(sample_dict)
        logging.info(f"Sample index stored in file {self.samples_index} ")
        total_samples = len(train_samples[0]) + len(test_samples[0])
        logging.info(
            f"Successfully stored dataset with train={len(train_samples[0])} + test={len(test_samples[0])} = {total_samples} "
            f"images in file {self.dataset_file} ({filesize_in_MB(self.dataset_file):.2f} MB)")

    def _create_dataset_dict(self, train_images, test_images):
        info = {"num_train": len(train_images[0]), "num_test": len(test_images[0])}
        return {"train": train_images, "test": test_images, "info": info}

    @staticmethod
    def load_from_folder(directory):
        """
        Creates dataset object and return
        :param directory:
        :return:
        """
        # read dataset info file
        # find dataset file
        dts_file = [os.path.join(directory, filename) for filename in os.listdir(directory) if
                    filename.endswith("pickle")]
        if dts_file is None or len(dts_file) != 1:
            raise ValueError("Dts folder must contain just one pickle file")
        dts_file = dts_file[0]

        dts = Dataset(directory)
        dts.dataset_file = dts_file
        dts_object = _load_dataset(dts.dataset_file)
        dts.info = dts_object["info"]

        train_dts, test_dts = dts_object["train"], dts_object["test"]
        x_train, y_train = train_dts
        x_test, y_test = test_dts

        logging.info(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
        logging.info(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
        logging.info("Original image shape : {}".format(x_train.shape))
        return x_train, y_train, x_test, y_test

    @staticmethod
    def load_from_file(dts_file):
        """
        Creates dataset object and return
        :param directory:
        :return:
        """
        dts_object = _load_dataset(dts_file)

        train_dts, test_dts = dts_object["train"], dts_object["test"]
        x_train, y_train = train_dts
        x_test, y_test = test_dts

        logging.info(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
        logging.info(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
        logging.info("Original image shape : {}".format(x_train.shape))
        return x_train, y_train, x_test, y_test

    ###################################
    ###### Dataset File
    ###################################

    def _save_dataset(self, dts):
        info = {"num_train": len(dts["train"][0]), "num_test": len(dts["test"][0])}
        dts["info"] = info  # recalculate
        with open(self.dataset_file, "wb") as f:
            pickle.dump(dts, f)

    ###################################
    ###### Sample info file
    ###################################

    def _load_sample_index(self):
        with open(self.samples_index, "r") as f:
            return json.load(f)

    def _save_sample_index(self, data):
        with open(self.samples_index, "w") as f:
            json.dump(data, f)

    ###################################
    ####### Extraction info
    ###################################

    def get_samples_info_by_idx(self, sample_index, test=True):
        """
        Returns original sample info looking by its position in dataset
        :return:
        """
        dts_object = _load_dataset(self.dataset_file)

        train_dts, test_dts = dts_object["train"], dts_object["test"]
        images, masks = test_dts if test else train_dts
        # find images and masks
        images = images[sample_index, ::]
        masks = masks[sample_index, ::]
        lst = []
        # load sample index to get original extraction id
        sample_idx_dict = self._load_sample_index()
        sample_idx_dict = sample_idx_dict["test"] if test else sample_idx_dict["train"]

        df = self.create_extractions_info()
        logging.info("Appending extraction info to samples")
        for i, sample_index in enumerate(sample_index):
            original_filename = sample_idx_dict[sample_index]
            extraction_id = self.get_extraction_id_for_sample(original_filename)
            ext_found = df[df.id == extraction_id]
            if ext_found.empty:
                raise ValueError(f"Couldn't find extraction info for sample: {original_filename}")
            ext_found = ext_found.values[0]
            image = (images[i] * 255).astype(np.uint8)
            lst.append({"image": image,
                        "mask": masks[i][..., 1],
                        "filename": original_filename,
                        "extraction": extraction_id,
                        "x": ext_found[1],
                        "y": ext_found[2]}
                       )
        return lst

    def create_extractions_info(self):
        """
        Gets raster folder and creates panda dataframe with info from each extraction to evaluate model results
        """
        centroid_list = []
        # open each extraction file and get raster centroid
        for filepath in self.get_extration_files():
            with rasterio.open(filepath) as src:
                # get raster coords
                centroid = src.bounds.left + (src.res[0] / 2), src.bounds.bottom + (src.res[1] / 2)
                # get extraction id from filename and create row info
                extraction_id = self.get_extraction_id_for_raster(filepath)
                centroid_list.append((extraction_id, centroid[0], centroid[1]))

        return pd.DataFrame(centroid_list, columns=["id", "x", "y"])

    def get_extraction_id_for_sample(self, sample_file):
        """
        Find original extraction file that includes the sample id
        :param id: filename with format {extraction_file}_{number}.jpeg, Ex: 0398_3-1_34_data.jpg
        :return:
        """
        # 0398_3-1_34_data.jpg --> 0398_3-1
        return "_".join(sample_file.split("_")[:-2])

    def get_extraction_id_for_raster(self, filepath):
        """
        Extractions are identified by the last part of the filename starting by __
        PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0425_2-1_0 -->0425_2-1_0
        :param filename:
        :return:
        """
        # PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0425_2-1_extraccion.tiff
        filename = os.path.basename(filepath)
        filename = os.path.splitext(filename)[0]
        parts = filename.split("__")
        if len(parts) != 2:
            raise ValueError(f"Invalid extraction filename: {filename} expected two parts separated by '__'")
        return parts[1]

    def get_sampling_info(self, sample_file):
        """
        Uses the extraction layer to get the pnoa tiles to sample and the number of images to extract from each tile.
        For each extraction a tuple is returned with the sampling info as a dict and the raster file to sample
        :param sample_file:
        :return:
        ({"0": 75, "1": 0, "mixed": 200}, '/absolute/path_to/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0345_6-4_0.tiff')
        """
        query = "select filename, n_samples_0, n_samples_1, n_samples_mixed from extractions"
        results = spt.list_all(sample_file, query)
        sampling_info = []
        for row in results:
            sinfo = {"0": row[1], "1": row[2], "mixed": row[3]}
            raster_path = os.path.join(self.extraction_folder, row[0])
            sampling_info.append((sinfo, raster_path))

        return sampling_info


def _load_dataset(dataset_file):
    with open(dataset_file, "rb") as f:
        return pickle.load(f)


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


def create_sample_arrays(image_dir, image_size, train_ratio=0.8, shuffle=True):
    """
    Crea un dataset para entrenar un modelo UNET a partir de imágenes y máscaras.

    Args:
        image_dir: La ruta al directorio que contiene las imágenes.
        mask_dir: La ruta al directorio que contiene las máscaras.
        image_size: El tamaño deseado de las imágenes (ancho, alto).
        batch_size: El tamaño del batch para entrenamiento.

    Returns:
        Un objeto tf.data.Dataset que contiene las imágenes y máscaras.
        :param train_ratio:
    """
    logging.info("Creating segmentation dataset")

    # Obtener la lista de nombres de archivo de las imágenes y máscaras
    image_names = sorted([f for f in os.listdir(image_dir) if "data" in f])
    mask_names = sorted([f for f in os.listdir(image_dir) if "mask" in f])

    assert len(image_names) == len(mask_names), "the list of imagen aren't same size"

    logging.info(
        f"{len(image_names)} images found. Shuffling={shuffle} and splitting with train_ratio = {train_ratio}")

    image_paths = [os.path.join(image_dir, name) for name in image_names]
    mask_paths = [os.path.join(image_dir, name) for name in mask_names]

    images, masks = create_array_dataset(image_paths, mask_paths, image_size)

    train_idx, test_idx = split_indexes(image_paths, train_ratio=train_ratio, shuffle=shuffle)

    train_dataset = np.stack([images[i] for i in train_idx], axis=0), np.stack([masks[i] for i in train_idx],
                                                                               axis=0)
    test_dataset = np.stack([images[i] for i in test_idx], axis=0), np.stack([masks[i] for i in test_idx], axis=0)
    sample_dict = {"train": [image_names[i] for i in train_idx], "test": [image_names[i] for i in test_idx]}

    return train_dataset, test_dataset, sample_dict


def create_array_dataset(image_paths, mask_paths, image_size):
    """
    Create dataset array from image and mask lists
    :param image_paths:
    :param mask_paths:
    :param image_size:
    :return:
    """
    images = []
    masks = []
    logging.info("Creating image array")
    for image_filename, mask_filename in zip(image_paths, mask_paths):
        sample = create_sample(image_filename, mask_filename, image_size=image_size)
        images.append(sample["image"])
        masks.append(sample["mask"])
    return images, masks


def create_tf_dataset(image_paths, mask_paths, batch_size):
    images = []
    masks = []
    for image_filename, mask_filename in zip(image_paths, mask_paths):
        sample = create_sample(image_filename, mask_filename)
        images.append(sample["image"])
        masks.append(sample["mask"])
    # create dataset from image and mask lists
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size)
    return dataset

    #######
    # creation of training datasets
    #######

    #
    # def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    #     assert (train_split + test_split + val_split) == 1
    #
    #     # Only allows for equal validation and test splits
    #     assert val_split == test_split
    #
    #     # Specify seed to always have the same split distribution between runs
    #     df_sample = df.sample(frac=1, random_state=12)
    #     indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]
    #
    #     train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    #
    #     return train_ds, val_ds, test_ds

    #
    # def display(display_list):
    #     plt.figure(figsize=(15, 15))
    #
    #     title = ['Input Image', 'True Mask', 'Predicted Mask']
    #
    #     for i in range(len(display_list)):
    #         plt.subplot(1, len(display_list), i + 1)
    #         plt.title(title[i])
    #         plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    #         plt.axis('off')
    #     plt.show()
    #
    #
    # class Augment(tf.keras.layers.Layer):
    #     def __init__(self, seed=42):
    #         super().__init__()
    #         # both use the same seed, so they'll make the same random changes.
    #         self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    #         self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    #
    #     def call(self, inputs, labels):
    #         inputs = self.augment_inputs(inputs)
    #         labels = self.augment_labels(labels)
    #         return inputs, labels


def split_indexes(dataset, train_ratio=0.6, test_ratio=0.2, validation_split=False, shuffle=True, seed=None):
    # Get the total number of examples in the dataset
    num_examples = len(dataset)

    # Get the index of the samples
    indexes = list(range(num_examples))

    # Optionally shuffle the indices
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indexes)

    # Calculate the sizes of each split
    train_size = int(train_ratio * num_examples)
    test_size = num_examples - train_size if validation_split is False else int(test_ratio * num_examples)
    val_size = 0 if validation_split is False else num_examples - train_size - test_size

    # Divide the indices into three groups for the splits
    train_indices = indexes[:train_size]
    test_indices = indexes[train_size:train_size + test_size]

    if val_size > 0:
        val_indices = indexes[train_size + test_size:train_size + test_size + val_size]

    if validation_split:
        return train_indices, test_indices, val_indices
    else:
        return train_indices, test_indices


def create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=64, shuffle=False):
    logging.info("Creating TF datasets from arrays")
    train_dts = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dts = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    if batch_size is not None:
        train_dts = train_dts.batch(batch_size)
        test_dts = test_dts.batch(batch_size)
    if shuffle:
        train_dts = train_dts.shuffle(len(x_train))
        test_dts = test_dts.shuffle(len(x_test))

    return train_dts, test_dts

    if __name__ == '__main__':
        dataset_folder = "/media/gus/data/viticola/datasets/segmenter/v4/"
        x_train, y_train, x_test, y_test = Dataset.load_from_folder(dataset_folder)
        train_dts, test_dts = create_tf_datasets(x_train, y_train, x_test, y_test)
        print(len(train_dts))
