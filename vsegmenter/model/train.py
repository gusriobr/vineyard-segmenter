import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from vsegmenter import cfg
from vsegmenter.data import dataset


def combineGenerator(gen1, gen2):
    while True:
        yield (gen1.next(), gen2.next())


def create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=64, shuffle=False):
    train_dts = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dts = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    if batch_size is not None:
        train_dts = train_dts.batch(batch_size)
        test_dts = test_dts.batch(batch_size)
    if shuffle:
        train_dts = train_dts.shuffle(len(x_train))
        test_dts = test_dts.shuffle(len(x_test))

    return train_dts, test_dts


def create_generators(x_train, y_train, x_test, y_test, batch_size=64, shuffle=True):
    train_gen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

    val_gen = ImageDataGenerator(rescale=1. / 255)

    args = dict(batch_size=batch_size, shuffle=shuffle)

    train_generator_image = train_gen.flow(x_train, **args)
    train_generator_mask = train_gen.flow(y_train, **args)
    train_generator = combineGenerator(train_generator_image, train_generator_mask)

    validation_generator_image = val_gen.flow(x_test, **args)
    validation_generator_mask = val_gen.flow(y_test, **args)
    validation_generator = combineGenerator(validation_generator_image, validation_generator_mask)
    return train_generator, validation_generator


if __name__ == '__main__':
    from model.unet_tf import unet_model

    ##### Load Dataset
    dataset_file = cfg.resource("dataset/v1_128.pickle")
    dts = dataset.load_dataset(dataset_file)
    train_dts, test_dts = dts["train"], dts["test"]
    x_train, y_train = train_dts
    x_test, y_test = test_dts
    print(f"Number of train samples: {len(x_train)}, test samples = {len(x_test)}")
    print(f"Sample dimensions: sample: {x_train[0].shape} label (mask): {y_train[0].shape}")
    print("Original image shape : {}".format(x_train.shape))

    ######## Create model

    OUTPUT_CLASSES = 2
    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    ######## Train model
    EPOCHS = 5
    VAL_SUBSPLITS = 2
    BATCH_SIZE = 64
    info = dts["info"]
    # num_examples =

    # train_generator, validation_generator = create_generators(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE)
    # train_steps = len(x_train) // BATCH_SIZE
    # val_steps = len(x_test) // BATCH_SIZE  # // VAL_SUBSPLITS
    # history = model.fit(train_generator, steps_per_epoch=train_steps, epochs=EPOCHS,
    #                     validation_data=validation_generator, validation_steps=val_steps)

    train_dts, test_dts = create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE)

    train_steps = len(x_train) // BATCH_SIZE
    val_steps = len(x_test) // BATCH_SIZE  # // VAL_SUBSPLITS

    model_history = model.fit(train_dts, epochs=EPOCHS,
                              steps_per_epoch=train_steps,
                              validation_steps=val_steps,
                              validation_data=test_dts)
