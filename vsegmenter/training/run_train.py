import logging
import os
from datetime import datetime

import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unet
from tensorflow import keras
from itertools import product
from data.dataset import Dataset, create_tf_datasets
from data.image import augment_image
from eval.evaluation import evaluate_on_dts
from image.augmentation import apply_imgaug_augmentation, tensorflow_imgaug_transform
from training.trainer import Trainer
from vsegmenter import cfg
from tensorflow.python.keras import backend as K

cfg.configLog()

if __name__ == '__main__':
    dataset_version = 7
    version = 6
    img_size = 128

    do_evaluate = False
    weights_file = None
    # weights_file = cfg.results('tmp/unet_6/2023-05-01T08-55_44')

    for apply_augmentation, dataset_version in product([False, True], [6, 7]):
        label = f"unet_dts{dataset_version}{'' if not apply_augmentation else '_aug'}"

        K.clear_session()
        tf.keras.backend.clear_session()

        dataset_file = cfg.dataset(f'v{dataset_version}/dataset_{img_size}.pickle')
        logging.info(f"Using dataset file: {dataset_file}")
        x_train, y_train, x_test, y_test = Dataset.load_from_file(dataset_file)

        x_train = x_train[:1000]
        y_train = y_train[:1000]

        logging.info(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
        logging.info(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
        logging.info("Original image shape : {}".format(x_train.shape))

        logging.info("--------------------------------------------------------------------")
        logging.info(f"Starting training with imag augmentation = {apply_augmentation}")

        train_dataset, validation_dataset = create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)
        if apply_augmentation:
            train_dataset = train_dataset.map(tensorflow_imgaug_transform)

        model_file = cfg.results(f"{label}_v{version}.model")
        history_file = cfg.results(f"{label}_v{version}_history.json")
        log_dir_path = cfg.results(f'tmp/{label}_{version}/{datetime.now().strftime("%Y-%m-%dT%H-%M_%S")}')

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
        if weights_file:
            logging.info(f"Training prev model_file with weights = {weights_file}")
            model.load_weights(weights_file)

        unet.finalize_model(model,
                            dice_coefficient=False,
                            auc=False)
        callbacks = [keras.callbacks.ModelCheckpoint(log_dir_path, save_weights_only=True,
                                                     save_best_only=True),
                     keras.callbacks.CSVLogger(log_dir_path + '_history.csv')
                     ]
        # trainer = unet.Trainer(checkpoint_callback=False, tensorboard_callback=False,
        #                        tensorboard_images_callback=False,
        #                        # learning_rate_scheduler=SchedulerType.WARMUP_LINEAR_DECAY,
        #                        callbacks=callbacks)
        trainer = Trainer(model, log_dir_path)

        EPOCHS = 100
        LEARNING_RATE = 1e-3
        batch_size = 32
        # history = trainer.fit(model, train_dataset, validation_dataset, epochs=EPOCHS, batch_size=batch_size)

        history = trainer.train(train_dataset.batch(batch_size), validation_dataset.batch(batch_size), epochs=EPOCHS,
                                batch_size=batch_size)
        logging.info(f"Training successfully finished. History file = {history_file}")

        if do_evaluate:
            dts = Dataset(cfg.dataset("v6"))
            logging.info(f"Evaluating model on dataset {dts}")
            tag = f"{label}_v{version}"
            output_folder = cfg.results(f"processed/v{version}")
            evaluate_on_dts(model, tag, dts, version, output_folder)

            logging.info(f"Evaluation finished")
