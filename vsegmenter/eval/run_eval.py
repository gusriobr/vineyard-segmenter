import json
import logging
import os

# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
import numpy as np
import tensorflow as tf
import unet

from data.dataset import Dataset
from vsegmenter import cfg

cfg.configLog()


def load_model(model_file):
    unet_model = unet.build_model(128, 128,
                                  channels=3,
                                  num_classes=2,
                                  layer_depth=5,
                                  filters_root=64,
                                  padding="same"
                                  )
    weights_file = os.path.join(model_file, "variables/variables")
    unet_model.load_weights(weights_file)
    unet.finalize_model(unet_model,
                        # loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        # metrics=[metrics.SparseCategoricalAccuracy()],
                        dice_coefficient=False,
                        auc=False)
    logging.info(f"Loadel model with weights = {weights_file}")
    return unet_model


if __name__ == '__main__':
    version = "v5"
    model_label = "unet"
    img_size = 128

    dataset_file = cfg.dataset(f'{version}/dataset_{img_size}.pickle')
    _, _, x_val, y_val = Dataset.load_from_file(dataset_file)

    x_val = x_val[:20]
    y_val = y_val[:20]

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    model = load_model(cfg.results("unet_v4.model"))

    y_pred = model.predict(x_val)
    pred_mask = tf.math.argmax(y_pred, axis=-1)
    pred_mask = pred_mask[..., np.newaxis]

    # detect the problems, get the top-20 images with errors
    bg_diff = np.abs(y_val[:, ..., 1], y_pred[:, ..., 1])
    sum_bg_diff_item_wise = np.sum(bg_diff, axis=(1, 2))
    ordered = np.argsort(-sum_bg_diff_item_wise)
    ordered = ordered[:20]

    # get original image files
    dts = Dataset(os.path.dirname(dataset_file))
    lst = dts.get_samples_info_by_idx(ordered)
    for i, index in enumerate(ordered):
        lst[i]["prediction"] = pred_mask[index][..., 0]

    # append mask as item

    results = model.evaluate(x_val, y_val, batch_size=128)
    logging.info(f"test loss, test acc: {results}")
    with open(cfg.results(f"{model_label}_evaluation.json"), "w") as f:
        json.dump(results, f)
