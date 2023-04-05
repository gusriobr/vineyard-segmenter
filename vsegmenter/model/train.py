import os

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
import unet
from unet import utils
from unet.datasets import circles
from model.train import create_tf_datasets
from vsegmenter import cfg
from vsegmenter.data import dataset
import json


def get_datasets():
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
    train_dataset, validation_dataset = get_datasets()
    version = "v2"
    model_file = cfg.results("segmenter_{version}.model")
    history_file = cfg.results("segmenter_{version}_history.json")

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

    EPOCHS = 50
    LEARNING_RATE = 1e-3
    history = trainer.fit(unet_model, train_dataset, validation_dataset, epochs=EPOCHS, batch_size=32)

    unet_model.save(model_file)

    with open(history_file, 'w') as f:
        json.dump(history.history, f)
