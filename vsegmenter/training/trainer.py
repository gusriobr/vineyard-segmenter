import os

import tensorflow as tf

K = tf.keras

NUM_CLASSES = 2


class Trainer:
    def __init__(self, model, log_dir, callbacks=None, scheduler=None, model_check_point=True,
                 add_tfb_callback=False, optimizer=None, opt_params=None, metrics=None):
        """
        Trainer class for training a Keras model.

        :param model: Keras model to be trained.
        :param log_dir: Directory for storing logs and model checkpoints.
        :param model_name: Name tag for the model. It will be used for naming the saved model checkpoint.
        :param callbacks: List of Keras callbacks to be used during training.
        :param scheduler: Learning rate scheduler function that takes an epoch index and returns a new learning rate.
        """
        self.model = model
        self.log_dir = log_dir
        self.callbacks = callbacks
        self.scheduler = scheduler
        self.add_tfb_callback = add_tfb_callback
        self.model_check_point = model_check_point
        self.optimizer = optimizer
        self.opt_params = opt_params
        self.metrics = metrics
        if not self.callbacks:
            self.callbacks = []
        if not self.opt_params:
            self.opt_params = {}

    def train(self, train_data, validation_data, epochs, batch_size):
        """
        Train the Keras model with the given data, using the specified callbacks, scheduler, and training parameters.

        :param train_data: Training data, can be an instance of tf.data.Dataset or any other format accepted by Keras model.fit.
        :param validation_data: Validation data, can be an instance of tf.data.Dataset or any other format accepted by Keras model.fit.
        :param epochs: Number of epochs to train the model for.
        :param batch_size: Number of samples per gradient update.
        :return: A Keras History object containing the training history.
        """
        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Add TensorBoard and ModelCheckpoint to the list of callbacks
        if self.add_tfb_callback:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
            self.callbacks.append(tensorboard)

        if self.model_check_point:
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.log_dir, "checkpoint.h5"), save_weights_only=True, save_best_only=True)
            self.callbacks.append(model_checkpoint)

        # Add the scheduler to callbacks if provided
        if self.scheduler is not None:
            self.callbacks.append(self.scheduler)

        self.callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(self.log_dir, "history.csv")))

        self._compile_model()

        # Train the model
        history = self.model.fit(train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size,
                                 callbacks=self.callbacks)

        return history

    def evaluate(self, test_data, batch_size):
        """
        Evaluate the Keras model on the given test data.

        :param test_data: Test data, can be an instance of tf.data.Dataset or any other format accepted by Keras model.evaluate.
        :param batch_size: Number of samples per evaluation batch.
        :return: A tuple (loss, metrics) containing the loss and metric values on the test data.
        """
        results = self.model.evaluate(test_data, batch_size=batch_size)
        return results

    def _compile_model(self):
        if self.optimizer is None:
            optimizer = K.optimizers.Adam(**self.opt_params)

        if self.metrics is None:
            metrics = ['categorical_crossentropy',
                       'categorical_accuracy',
                       mean_iou
                       ]

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)




def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.dtypes.float64)
    y_pred = tf.cast(y_pred, tf.dtypes.float64)
    I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tf.reduce_mean(I / U)