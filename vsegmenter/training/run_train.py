import logging
from datetime import datetime

import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unet
from tensorflow import keras
from tensorflow.python.keras import backend as K

from data.dataset import Dataset, create_tf_datasets
from eval.evaluation import evaluate_on_dts
from image.augmentation import make_transformer, get_imgaug
from training.trainer import Trainer
from vsegmenter import cfg

cfg.configLog()

if __name__ == '__main__':
    dataset_version = 7
    version = 7
    img_size = 128

    dataset_file = cfg.dataset(f'v{dataset_version}/dataset_{img_size}.pickle')
    logging.info(f"Using dataset file: {dataset_file}")
    x_train, y_train, x_test, y_test = Dataset.load_from_file(dataset_file)

    logging.info(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
    logging.info(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
    logging.info("Original image shape : {}".format(x_train.shape))

    do_evaluate = True
    apply_augmentation = True
    weights_file = None
    # weights_file = cfg.results('tmp/unet_6/2023-05-01T08-55_44')
    # weights_file = cfg.results('tmp/unet_7/2023-05-11T15-54_17')
    apply_augmentation = True
    EPOCHS = 3000
    for lr in [5e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
        aug_type = "noise"
        label = f"unet_{version}_exp_{lr}"

        K.clear_session()
        tf.keras.backend.clear_session()

        logging.info("--------------------------------------------------------------------")
        logging.info(f"Starting training with imag augmentation = {apply_augmentation}")

        train_dataset, validation_dataset = create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)
        if apply_augmentation:
            aug_sequences = get_imgaug(aug_type)
            aug_transformer = make_transformer(*aug_sequences)
            train_dataset = train_dataset.map(lambda image, mask: tf.py_function(aug_transformer, [image, mask],
                                                                                 [tf.float32, tf.float32]))

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

        LEARNING_RATE = lr
        batch_size = 32
        # history = trainer.fit(model, train_dataset, validation_dataset, epochs=EPOCHS, batch_size=batch_size)

        history = trainer.train(train_dataset.batch(batch_size), validation_dataset.batch(batch_size), epochs=EPOCHS,
                                batch_size=batch_size)
        logging.info(f"Training successfully finished. History file = {history_file}")

        weights_file = log_dir_path

        if do_evaluate:
            dts = Dataset(cfg.dataset(f"v{dataset_version}"))
            logging.info(f"Evaluating model on dataset {dts}")
            tag = f"{label}_v{version}" if not apply_augmentation else f"{label}_v{version}_imgaug"
            output_folder = cfg.results(f"processed/{label}")
            evaluate_on_dts(model, tag, dts, version, output_folder)

            logging.info(f"Evaluation finished")
