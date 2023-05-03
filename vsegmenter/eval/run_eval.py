import json
import logging
import os

# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
import numpy as np
import tensorflow as tf
import unet

from data.dataset import Dataset
from eval.evaluation import evaluate_on_dts
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
    version = 6
    model_label = "unet"
    img_size = 128

    tag = f"{model_label}_v{version}"
    dts = Dataset(cfg.dataset(f'v{version}'))

    output_folder = cfg.results(f"processed/v{version}")
    weights_file = cfg.results('tmp/unet_6/2023-05-01T08-55_44')
    logging.info(f"Loading file using weights file {weights_file}")
    unet_model = build_model(weights_file)
    evaluate_on_dts(unet_model, tag, dts, version, output_folder)


