import json
import os
from datetime import datetime

# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
import tensorflow as tf
import unet
from tensorflow import keras

from unet import custom_objects
from tensorflow.python.keras import backend as K
from unet.schedulers import SchedulerType
from data.dataset import create_tf_datasets
from vsegmenter import cfg
from vsegmenter.data import dataset

import logging

cfg.configLog()


def get_datasets(version):
    dataset_file = cfg.dataset(f'{version}/dataset_128.pickle')
    dts = dataset.load_dataset(dataset_file)
    train_dts, test_dts = dts["train"], dts["test"]
    x_train, y_train = train_dts
    x_test, y_test = test_dts
    # x_train = x_train[0:32]
    # y_train = y_train[0:32]
    # x_test = x_test[0:32]
    # y_test = y_test[0:32]
    logging.info(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
    logging.info(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
    logging.info("Original image shape : {}".format(x_train.shape))
    return create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)


if __name__ == '__main__':
    train_dataset, validation_dataset = get_datasets(version="v4")

    version = "v4"
    label = "unet"
    model_file = cfg.results(f"{label}_{version}.model")
    history_file = cfg.results(f"{label}_{version}_history.json")
    log_dir_path = cfg.results(f'tmp/{label}/{datetime.now().strftime("%Y-%m-%dT%H-%M_%S")}')

    prev_model = cfg.results(f"unet_v3_a.model")
    logging.info(f"model_file = {model_file}")
    logging.info(f"history_file = {history_file}")
    logging.info(f"log_dir_path = {log_dir_path}")

    unet_model = unet.build_model(128, 128,
                                  channels=3,
                                  num_classes=2,
                                  layer_depth=5,
                                  filters_root=64,
                                  padding="same"
                                  )
    if prev_model:
        weights_file = os.path.join(prev_model, "variables/variables")
        logging.info(f"Training prev model_file with weights = {weights_file}")
        unet_model.load_weights(weights_file)

    unet.finalize_model(unet_model,
                        # loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        # metrics=[metrics.SparseCategoricalAccuracy()],
                        dice_coefficient=False,
                        auc=False)

    callbacks = [keras.callbacks.ModelCheckpoint(log_dir_path, save_weights_only=True,
                                                 save_best_only=True),
                 keras.callbacks.CSVLogger(log_dir_path + '_history.csv')
                 ]
    trainer = unet.Trainer(checkpoint_callback=False,
                           tensorboard_callback=False,
                           tensorboard_images_callback=False,
                           # learning_rate_scheduler=SchedulerType.WARMUP_LINEAR_DECAY,
                           callbacks=callbacks)

    EPOCHS = 200
    LEARNING_RATE = 1e-3
    history = trainer.fit(unet_model, train_dataset, validation_dataset, epochs=EPOCHS, batch_size=32)

    # unet_model.save(model_file)
    # unet_model.save_weights(model_file + ".h5")

    # with open(history_file, 'w') as f:
    #     json.dump(history.history, f)
