import logging
import os
from datetime import datetime

import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unet
from tensorflow import keras

from data.dataset import Dataset, create_tf_datasets
from data.image import augment_image
from training.trainer import Trainer
from vsegmenter import cfg

cfg.configLog()

if __name__ == '__main__':
    version = "v5"
    label = "unet"
    img_size = 128

    dataset_file = cfg.dataset(f'{version}/dataset_{img_size}.pickle')
    logging.info(f"Using dataset file: {dataset_file}")
    x_train, y_train, x_test, y_test = Dataset.load_from_file(dataset_file)

    logging.info(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
    logging.info(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
    logging.info("Original image shape : {}".format(x_train.shape))

    train_dataset, validation_dataset = create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)
    # train_dataset = train_dataset.map(lambda x, y: (tf.py_function(augment_image, [x, y], [tf.float16, tf.uint8])))

    model_file = cfg.results(f"{label}_{version}.model")
    history_file = cfg.results(f"{label}_{version}_history.json")
    log_dir_path = cfg.results(f'tmp/{label}/{datetime.now().strftime("%Y-%m-%dT%H-%M_%S")}')

    prev_model = None # cfg.results(f"unet_v3_a.model")
    logging.info(f"model_file = {model_file}")
    logging.info(f"history_file = {history_file}")
    logging.info(f"log_dir_path = {log_dir_path}")

    model = unet.build_model(128, 128,
                             channels=3,
                             num_classes=2,
                             layer_depth=5,
                             filters_root=64,
                             padding="same"
                             )
    if prev_model:
        weights_file = os.path.join(prev_model, "variables/variables")
        # weights_file = '/workspaces/wml/vineyard-segmenter/results/tmp/unet/2023-04-27T16-19_53'
        # weights_file = '/workspaces/wml/vineyard-segmenter/results/unet_v4.model/variables/variables'
        logging.info(f"Training prev model_file with weights = {weights_file}")
        model.load_weights(weights_file)

    unet.finalize_model(model,
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

    EPOCHS = 2000
    LEARNING_RATE = 1e-3
    batch_size = 32
    history = trainer.fit(model, train_dataset, validation_dataset, epochs=EPOCHS, batch_size=batch_size)

    # history = trainer.train(train_dataset.batch(batch_size), validation_dataset.batch(batch_size), epochs=EPOCHS, batch_size=batch_size)
    logging.info(f"Training successfully finished. History file = {history_file}")
