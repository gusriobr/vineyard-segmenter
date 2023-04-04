import matplotlib.pyplot as plt
import numpy as np
import unet
from unet import utils
from unet.datasets import circles
from tensorflow.keras import losses, metrics
from model.train import create_tf_datasets
from vsegmenter import cfg
from vsegmenter.data import dataset
import tensorflow as tf
from unet import custom_objects

import pickle

def get_datasets():
    ##### Load Dataset
    dataset_file = cfg.resource("dataset/v1_128.pickle")
    dts = dataset.load_dataset(dataset_file)
    train_dts, test_dts = dts["train"], dts["test"]
    x_train, y_train = train_dts
    x_test, y_test = test_dts
    print(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
    print(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
    print("Original image shape : {}".format(x_train.shape))
    return create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)


if __name__ == '__main__':
    train_dataset, validation_dataset = circles.load_data(100, nx=200, ny=200, splits=(0.7, 0.3))
    train_dataset, validation_dataset = get_datasets()
    history_file = cfg.results("modelv1_history.pickle")
    model_file = cfg.results("segmenter_v1.model")

    unet_model = unet.build_model(128, 128,
                                  channels=3,
                                  num_classes=2,
                                  layer_depth=5,
                                  filters_root=64,
                                  padding="same"
                                  )

    unet.finalize_model(unet_model,
                        # loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        # metrics=[metrics.SparseCategoricalAccuracy()],
                        dice_coefficient=False,
                        auc=False)

    # unet.finalize_model(unet_model,
    #                     loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    #                     # metrics=[metrics.SparseCategoricalAccuracy()],
    #                     auc=False,
    #                     learning_rate=LEARNING_RATE)

    trainer = unet.Trainer(checkpoint_callback=False, tensorboard_callback=False, tensorboard_images_callback=False)

    LEARNING_RATE = 1e-3
    history = trainer.fit(unet_model, train_dataset, validation_dataset, epochs=25, batch_size=64)

    unet_model.save(model_file)

    with open(history_file, "wb") as hfile:
        pickle.dump(history, hfile)

