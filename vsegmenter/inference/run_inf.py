import glob
import logging
import os
import sys

import cv2
import numpy as np
import skimage.io
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from skimage import io
from tensorflow.python.keras import backend as K

from image.raster import georeference_image
from vsegmenter import cfg
from image.sliding import batched_sliding_window

ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)

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


#
# def predict(model, images):
#     prediction = unet_model.predict(validation_images)
#     if std_f is not None:
#         x = std_f(x)
#         return model.predict(x)
#     else:
#         predict_gen = datagen.flow(x, y=None, shuffle=False)
#         return model.predict_generator(predict_gen)


def apply_model_slide(image_path, model, window_size=(48, 48), step_size=48, batch_size=64, threshold=0.8):
    # load the input image
    image = read_img(image_path)

    # grab the dimensions of the input image and crop the image such
    # that it tiles nicely when we generate the training data +
    # labels
    (h, w) = image.shape[:2]

    output_img = np.zeros((h, w, 1), dtype=np.uint8)

    for images, positions in batched_sliding_window(image, window_size, step_size, batch_size=batch_size):
        # apply model
        img_rescaled = np.empty((len(images), 128, 128, 3))
        for i in range(len(images)):
            # Redimensiona la imagen utilizando cv2.resize()
            img_rescaled[i] = cv2.resize(images[i], (128, 128), interpolation=cv2.INTER_AREA)

        y_pred = model.predict(img_rescaled)
        # create mask from model preditions
        pred_mask = tf.math.argmax(y_pred, axis=-1)

        # paste into original image
        for i, pos in enumerate(positions):
            rescaled = cv2.resize(pred_mask[i].numpy(), (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
            rescaled = rescaled[..., np.newaxis]
            output_img[pos[1]:pos[1] + window_size[0], pos[0]:pos[0] + window_size[1]] = rescaled

    return output_img


def read_img(path):
    rimg = io.imread(path)
    return rimg


def build_model(model_folder, img_size=48):
    """
    :param model_folder:
    :param img_size:
    :return:
    """
    from unet import custom_objects
    return tf.keras.models.load_model(model_folder, custom_objects=custom_objects)


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
        [os.path.join(model_folder, 'segmenter_v1.model'), 'unet_base', 1],
    ]
    input_folder = '/media/cartografia/01_Ortofotografia/'
    output_folder = '/media/gus/data/viticola/segmentation'

    # find all nested images
    # input_images = load_pnoa_filenames(input_folder, cfg.project_file('vineyard/inference/pnoa_files.txt'))
    # input_images.sort()

    input_images = [
        '/media/gus/workspace/wml/vineyard-segmenter/resources/raster/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0373_7-2_extraction.tiff']
    # input_images = [
    #     '/media/gus/data/rasters/aerial/pnoa/2020/H50_0373/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0373_7-2.tif']

    patch_size = 256
    for m in models:
        # clear tensorflow memory
        K.clear_session()
        tf.keras.backend.clear_session()
        model_path = m[0]
        tag = m[1]  # model_name

        model = build_model(model_path)

        total = len(input_images)
        for idx, input in enumerate(input_images):
            logging.info("Processing image {} of {} - {}".format(idx, total, input))
            filename = os.path.basename(input)
            base, ext = os.path.splitext(filename)
            outf = os.path.join(output_folder, "{}_{}{}".format(base, tag, ext))
            output_img = apply_model_slide(input, model, window_size=(patch_size, patch_size), step_size=patch_size,
                                           batch_size=1024 * 10)
            skimage.io.imsave(outf, output_img)

            # apply_model(input, patch_size, outf, model, std_f=std_func, channels=[0, 1, 2], scale=1)

            logging.info("Applying geolocation info.")
            rimg = read_img(outf)
            if len(rimg.shape) == 2:
                rimg = rimg[:, :, np.newaxis]
            georeference_image(rimg, input, outf, scale=1, bands=1)
            logging.info("Finished processing file {}, \ngenerated output raster {}.".format(input, outf))

    # plt.show()
    logging.info("========================================")
    logging.info("Model inference  on raster finished.")
    logging.info("========================================")
