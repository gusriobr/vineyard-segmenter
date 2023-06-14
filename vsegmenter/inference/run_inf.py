import argparse
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

from data.dataset import Dataset
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
        # create mask from model predictions
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


def build_model(weights_file=None, img_size=48):
    """
    :param model_folder:
    :param img_size:
    :return:
    """
    import unet
    from unet import custom_objects
    if weights_file:
        unet_model = unet.build_model(128, 128,
                                      channels=3,
                                      num_classes=2,
                                      layer_depth=5,
                                      filters_root=64,
                                      padding="same"
                                      )
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
    logging.info(f"Listing pnoa tiles from file {tile_file} using base folder {base_folder}")

    lines = open(tile_file).read().splitlines()
    files = set()
    for line in lines:
        if not line:  # empty line
            continue
        # is a file
        fabs = line if os.path.isabs(line) else f"{base_folder}/{line}"
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


def is_cuda_disabled():
    cuda_env_var = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    return not cuda_env_var


def get_input_images(args):
    if args.run_on_dts:
        # by default use extraction files from dataset
        dts = Dataset(dts_folder='/media/gus/data/viticola/datasets/segmenter/v5_ribera')
        input_images = dts.get_extration_files()
    else:
        pnoa_index_file = args.pnoa_index_file
        # find all nested images
        if pnoa_index_file is None:
            # default index file for testing
            pnoa_index_file = cfg.project_file('vsegmenter/inference/pnoa_files.txt')
            # pnoa_index_file = cfg.resources('pnoa_ribera.txt')
            pnoa_index_file = cfg.resources('pnoa_arlanza.txt')
        logging.info(f"Using pnoa index as input: {pnoa_index_file}")
        input_images = load_pnoa_filenames(cfg.PNOA_BASE_FOLDER, pnoa_index_file)

    return input_images


def predict_on_raster(raster_file, model, output_file, scale_factor=0.5, patch_size=128, sliding_batch_size=1024):
    image = io.imread(raster_file)
    original_shape = image.shape
    if scale_factor != 1:
        image = down_img(image, scale=scale_factor)

    # apply model to rescaled image
    logging.info("Applying model to image")
    img_prediction = apply_model_slide(image, model, window_size=(patch_size, patch_size), step_size=patch_size,
                                       batch_size=sliding_batch_size)
    skimage.io.imsave(output_file, img_prediction)
    logging.info("Applying geolocation info")
    if len(img_prediction.shape) == 2:
        img_prediction = img_prediction[:, :, np.newaxis]
    # rescale image to original raster size
    if scale_factor != 1:
        img_prediction = up_img(img_prediction, (original_shape[0], original_shape[1], 1), order=1)
    georeference_image(img_prediction, raster_file, output_file, scale=1, bands=1)
    logging.info("Finished processing file {}, \ngenerated output raster {}.".format(raster_file, output_file))


if __name__ == '__main__':
    logging.info(f"IS_CUDA_DISABLED = : {is_cuda_disabled()}")

    """
    Ejemplo lanzamiento
    nohup ./run_inference.sh 7 --pnoa_index_file /workspaces/wml/vineyard-segmenter/resources/pnoa_leon.txt --output_folder /workspaces/wml/vineyard-segmenter/results/processed/v7_leon &
    """
    parser = argparse.ArgumentParser(description="Running inference con vsegmentation models")
    parser.add_argument("version", type=int, help="Model version")
    parser.add_argument("--run_on_dts", type=bool, help="Run model on extraction files", default=False, required=False)
    parser.add_argument("--pnoa_index_file", type=str, help="Fichero que contiene el listado de hojas pnoa a procesar",
                        default=None, required=False)
    parser.add_argument("--model_file", type=str,
                        help="Fichero del modelo a utilizar, ej: absolutepath/tmp/unet_v7/2023-05-24T04-24_52",
                        default=None, required=False)
    parser.add_argument("--output_folder", type=str,
                        help="Ruta destino de los archivos generados, por defecto results/processed/v{version}",
                        default=None, required=False)
    args = parser.parse_args()

    version = args.version
    model_folder = cfg.results("")

    models = [
        # [os.path.join(model_folder, f"unet_v{version}.model"), f'unet_v{version}'],
        # [f'unet_v{version}', cfg.results('tmp/unet_6/2023-05-01T08-55_44')],
        [f'unet_v{version}', cfg.results('unet_v7/2023-05-24T04-24_52')],
    ]
    output_folder = cfg.results(f"processed/v{version}")
    if args.output_folder is not None:
        output_folder = args.output_folder
    logging.info(f"Output folder for processed raster masks {output_folder}")

    input_images = get_input_images(args)
    logging.info(f"Number of input images to process: {len(input_images)}")

    patch_size = 128
    scale_factor = 0.5
    SLIDING_BATCH_SIZE = 2048
    logging.info(f"SLIDING_BATCH_SIZE: {SLIDING_BATCH_SIZE}")
    # input_images = input_images[:2]
    for m in models:
        logging.info(f"Running model: {m}")
        # clear tensorflow memory
        K.clear_session()
        tf.keras.backend.clear_session()
        tag = m[0]  # model_name
        weights_file = m[1]
        unet_model = build_model(weights_file)

        total = len(input_images)
        for idx, raster_file in enumerate(input_images):
            logging.info("Processing image {} of {} - {}".format(idx + 1, total, raster_file))
            filename = os.path.basename(raster_file)
            base, ext = os.path.splitext(filename)
            outf = os.path.join(output_folder, "{}_{}{}".format(base, tag, ext))

            # load and scale the input image
            predict_on_raster(raster_file, unet_model, outf, patch_size=patch_size, scale_factor=scale_factor,
                              sliding_batch_size=SLIDING_BATCH_SIZE)

    # plt.show()
    logging.info("========================================")
    logging.info("Model inference on raster files finished.")
    logging.info("========================================")
