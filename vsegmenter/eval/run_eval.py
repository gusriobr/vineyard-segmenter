import json
import logging
import os

# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
import numpy as np
import tensorflow as tf
import unet

from data.dataset import Dataset
from inference.run_inf import predict_on_raster, build_model
from postproc.run_post import post_process_images
from vsegmenter import cfg

cfg.configLog()


def load_model(weights_file=None):
    unet_model = unet.build_model(128, 128,
                                  channels=3,
                                  num_classes=2,
                                  layer_depth=5,
                                  filters_root=64,
                                  padding="same"
                                  )
    if weights_file:
        unet_model.load_weights(weights_file)
    unet.finalize_model(unet_model,
                        # loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        # metrics=[metrics.SparseCategoricalAccuracy()],
                        dice_coefficient=False,
                        auc=False)
    logging.info(f"Loadel model with weights = {weights_file}")
    return unet_model


if __name__ == '__main__':
    version = 5
    model_label = "unet"
    img_size = 128

    tag = f"{model_label}_v{version}"
    dts = Dataset(cfg.dataset(f'v{version}'))

    output_folder = cfg.results(f"processed/v{version}")
    weights_file = '/media/gus/workspace/wml/vineyard-segmenter/results/unet_v5/2023-04-29T23-00_32'
    logging.info(f"Loading file using weights file {weights_file}")
    unet_model = build_model(weights_file)

    # runs model on dataset extractions to evaluation the modelo on dataset
    raster_files = dts.get_extration_files()
    total = len(raster_files)
    output_raster_list = []
    for idx, r_file in enumerate(raster_files):
        logging.info("Processing image {} of {} - {}".format(idx + 1, total, r_file))
        filename = os.path.basename(r_file)
        base, ext = os.path.splitext(filename)
        output_file = os.path.join(output_folder, "{}_{}{}".format(base, tag, ext))

        # predict_on_raster(r_file, unet_model, output_file)
        output_raster_list.append(output_file)

    logging.info(f"Prediction on dataset finished. Starting post-processing")
    output_db_file = cfg.results(f"processed/v{version}/polygons_v{version}.sqlite")
    post_process_images(output_raster_list, output_db_file)
    logging.info(f"Evaluation finished output_folder: {output_folder}")

