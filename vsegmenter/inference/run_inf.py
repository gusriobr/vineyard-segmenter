import glob
import logging
import os
import sys

import numpy as np
import skimage.io
import tensorflow as tf
from skimage import io
from skimage.transform import resize
from tensorflow.python.keras import backend as K

from image.raster import georeference_image
from image.sliding import batched_sliding_window
from vsegmenter import cfg

# ROOT_DIR = os.path.abspath("../../../")
# sys.path.append(ROOT_DIR)

layers = tf.keras.layers

cfg.configLog()


def comp_file(filename):
    return os.path.join(cfg.root_folder(), filename)


def get_file(filepath):
    return cfg.file_path(filepath)
    # basepath = os.path.dirname(os.path.abspath(__file__))
    # return os.path.join(basepath, filepath)


def get_folder(filepath):
    f_path = cfg.file_path(filepath)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    return f_path


def pred_to_category(value, threshold=0.80):
    v = 255 if value >= threshold else 0
    return np.array([v], dtype=np.uint8)


def standarize(mean, std):
    def f_inner(x):
        x_tr = x * (1.0 / 255.0)
        x_tr -= mean
        x_tr /= std
        return x_tr

    return f_inner


def apply_model_slide(image, model, window_size=(48, 48), step_size=48, batch_size=64, scale_factor=1):
    # grab the dimensions of the input image and crop the image such
    # that it tiles nicely when we generate the training data +
    # labels
    (h, w) = image.shape[:2]

    output_img = np.zeros((h, w, 1), dtype=np.uint8)

    for images, positions in batched_sliding_window(image, window_size, step_size, batch_size=batch_size):
        # img_rescaled = np.empty((len(images), 128, 128, 3))
        # for i in range(len(images)):
        #     # Redimensiona la imagen utilizando cv2.resize()
        #     img_rescaled[i] = cv2.resize(images[i], (128, 128), interpolation=cv2.INTER_AREA)

        # apply model
        y_pred = model.predict(images)
        # create mask from model preditions
        pred_mask = tf.math.argmax(y_pred, axis=-1)
        pred_mask = pred_mask[..., np.newaxis]
        # paste into original image
        for i, pos in enumerate(positions):
            # rescaled = cv2.resize(pred_mask[i].numpy(), (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
            # rescaled = rescaled[..., np.newaxis]
            output_img[pos[1]:pos[1] + window_size[0], pos[0]:pos[0] + window_size[1]] = pred_mask[i].numpy()

    return output_img


def down_img(img, scale=1):
    if scale == 1:
        return img
    height, width, _ = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    img = resize(img, (new_height, new_width))
    return img


def up_img(img, shape, order, threshold=0.5):
    """
    rescale image to given shape keeping the image as 0-1 np.uint8
    :param img:
    :param shape:
    :param order: order of interpolation:
            0: Nearest-neighbor
            1: Bi-linear (default)
            2: Bi-quadratic
            3: Bi-cubic
            4: Bi-quartic
            5: Bi-quintic
    :return:
    """
    img *= 255
    img = resize(img, (shape[0], shape[1]), order=order)
    return np.where(img >= threshold, 1, 0).astype(np.uint8)


def build_model(model_folder, load_from_weights=True, img_size=48):
    """
    :param model_folder:
    :param img_size:
    :return:
    """
    import unet
    from unet import custom_objects
    if load_from_weights:
        unet_model = unet.build_model(128, 128,
                                      channels=3,
                                      num_classes=2,
                                      layer_depth=5,
                                      filters_root=64,
                                      padding="same"
                                      )
        weights_file = os.path.join(model_folder, "variables/variables")
        unet_model.load_weights(weights_file)
    else:
        unet_model = tf.keras.models.load_model(model_folder, custom_objects=custom_objects)
    return unet_model


def load_pnoa_filenames(base_folder, tile_file):
    """
    Localiza tiles del pnoa a partir de fichero
    :param base_folder:
    :return:
    """
    lines = open(tile_file).read().splitlines()
    files = set()
    for line in lines:
        if line:  # not empty
            # is a file
            fabs = "{}/{}"
            if os.path.exists(fabs) and os.path.isfile(fabs):
                files.add(fabs)
            else:
                nested_files = glob.glob("{}/{}/*.tif".format(base_folder, line))
                if len(nested_files) > 0:
                    # it has nested tif
                    files.update(nested_files)
                else:
                    # it has nested folders with tifs
                    nested_files = glob.glob("{}/{}/**/*.tif".format(base_folder, line))
                    files.update(nested_files)
    lst_files = list(files)
    lst_files.sort()
    return lst_files


if __name__ == '__main__':
    # load srs model

    model_folder = cfg.results("")
    models = [
        [os.path.join(model_folder, 'unet_v4.model'), 'unet_v4'],
    ]
    input_folder = '/media/cartografia/01_Ortofotografia/'
    output_folder = '/media/gus/data/viticola/segmentation'

    # find all nested images
    # input_images = load_pnoa_filenames(input_folder, cfg.project_file('vineyard/inference/pnoa_files.txt'))
    # input_images.sort()

    dataset_version = "v4"
    base_folder = cfg.dataset(f"{dataset_version}/extractions")
    input_images = [os.path.join(base_folder, raster_file) for raster_file in os.listdir(base_folder) if
                    os.path.isfile(os.path.join(base_folder, raster_file))]

    patch_size = 128
    scale_factor = 0.5
    for m in models:
        # clear tensorflow memory
        K.clear_session()
        tf.keras.backend.clear_session()
        model_path = m[0]
        tag = m[1]  # model_name

        model = build_model(model_path)

        total = len(input_images)
        for idx, raster_file in enumerate(input_images):
            logging.info("Processing image {} of {} - {}".format(idx + 1, total, raster_file))
            filename = os.path.basename(raster_file)
            base, ext = os.path.splitext(filename)
            outf = os.path.join(output_folder, "{}_{}{}".format(base, tag, ext))

            # load and scale the input image
            image = io.imread(raster_file)
            original_shape = image.shape
            image = down_img(image, scale=scale_factor)

            # apply model to rescaled image
            logging.info("Applying model to image")
            img_prediction = apply_model_slide(image, model, window_size=(patch_size, patch_size), step_size=patch_size,
                                               batch_size=1024 * 10)
            skimage.io.imsave(outf, img_prediction)

            logging.info("Applying geolocation info")
            # rimg = read_img(outf)
            if len(img_prediction.shape) == 2:
                img_prediction = img_prediction[:, :, np.newaxis]
            # rescale image to original raster size
            img_prediction = up_img(img_prediction, (original_shape[0], original_shape[1], 1), order=1)

            georeference_image(img_prediction, raster_file, outf, scale=1, bands=1)
            logging.info("Finished processing file {}, \ngenerated output raster {}.".format(raster_file, outf))

    # plt.show()
    logging.info("========================================")
    logging.info("Model inference  on raster finished.")
    logging.info("========================================")
