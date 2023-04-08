import os

from data.dataset import create_tf_datasets

# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
import unet
from vsegmenter import cfg
from vsegmenter.data import dataset
import json
import tensorflow as tf


def get_datasets(version):
    dataset_file = cfg.dataset(f'{version}/dataset_128.pickle')
    dts = dataset.load_dataset(dataset_file)
    train_dts, test_dts = dts["train"], dts["test"]
    x_train, y_train = train_dts
    x_test, y_test = test_dts
    print(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
    print(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
    print("Original image shape : {}".format(x_train.shape))
    return create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=None)


if __name__ == '__main__':
    train_dataset, validation_dataset = get_datasets(version="v3")

    version = "v3"
    model_file = cfg.results(f"segmenter_{version}.model")
    history_file = cfg.results(f"segmenter_{version}_history.json")

    prev_model = cfg.results(f"segmenter_v2.model")
    if prev_model:
        from unet import custom_objects

        unet_model = tf.keras.models.load_model(prev_model, custom_objects=custom_objects)
    else:
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

    trainer = unet.Trainer(checkpoint_callback=True, tensorboard_callback=False, tensorboard_images_callback=False)

    EPOCHS = 200
    LEARNING_RATE = 1e-3
    history = trainer.fit(unet_model, train_dataset, validation_dataset, epochs=EPOCHS, batch_size=32)

    unet_model.save(model_file)

    with open(history_file, 'w') as f:
        json.dump(history.history, f)
