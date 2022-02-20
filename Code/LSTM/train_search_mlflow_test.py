"""
Train a simple Keras DL model on the dataset used in MLflow tutorial (wine-quality.csv).

Dataset is split into train (~ 0.56), validation(~ 0.19) and test (0.25).
Validation data is used to select the best hyperparameters, test set performance is evaluated only
at epochs which improved performance on the validation dataset. The model with best validation set
performance is logged with MLflow.
"""
import warnings

import math

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

import click

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.keras
import mlflow.tensorflow

import argparse
import config
import os
import random
import utils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", required=False, help="path to output loss plot")
args = vars(ap.parse_args())

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def eval_and_log_metrics(prefix, actual, pred, epoch):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred)
    mlflow.log_metric("{}_rmse".format(prefix), rmse, step=epoch)
    mlflow.log_metric("{}_mape".format(prefix), mape, step=epoch)
    return rmse, mape


def get_standardize_f(train):
    mu = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    return lambda x: (x - mu) / std


class MLflowCheckpoint(Callback):
    """
    Example of Keras MLflow logger.
    Logs training metrics and final model with MLflow.

    We log metrics provided by Keras during training and keep track of the best model (best loss
    on validation dataset). Every improvement of the best model is also evaluated on the test set.

    At the end of the training, log the best model with MLflow.
    """

    def __init__(self, test_x, test_y, loss="rmse"):
        self._test_x = test_x
        self._test_y = test_y
        self.train_loss = "train_{}".format(loss)
        self.val_loss = "val_{}".format(loss)
        self.test_loss = "test_{}".format(loss)
        self._best_train_loss = math.inf
        self._best_val_loss = math.inf
        self._best_model = None
        self._next_step = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log the best model at the end of the training run.
        """
        if not self._best_model:
            raise Exception("Failed to build any model")
        mlflow.log_metric(self.train_loss, self._best_train_loss, step=self._next_step)
        mlflow.log_metric(self.val_loss, self._best_val_loss, step=self._next_step)
        mlflow.keras.log_model(self._best_model, "model")

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. If model improved on the validation data, evaluate it on
        a test set and store it as the best model.
        """
        if not logs:
            return
        self._next_step = epoch + 1
        train_loss = logs["loss"]
        val_loss = logs["val_loss"]
        mlflow.log_metrics({self.train_loss: train_loss, self.val_loss: val_loss}, step=epoch)

        if val_loss < self._best_val_loss:
            # The result improved in the validation set.
            # Log the model with mlflow and also evaluate and log on test set.
            self._best_train_loss = train_loss
            self._best_val_loss = val_loss
            self._best_model = keras.models.clone_model(self.model)
            self._best_model.set_weights([x.copy() for x in self.model.get_weights()])
            preds = self._best_model.predict(self._test_x)
            eval_and_log_metrics("test", self._test_y, preds, epoch)


@click.command(
    help="Trains an Keras model on wine-quality dataset."
    "The input is expected in csv format."
    "The model and its metrics are logged with mlflow."
)
@click.option("--epochs", type=click.INT, default=100, help="Maximum number of epochs to evaluate.")
@click.option(
    "--batch-size", type=click.INT, default=16, help="Batch size passed to the learning algo."
)
@click.option("--learning-rate", type=click.FLOAT, default=1e-2, help="Learning rate.")
@click.option("--momentum", type=click.FLOAT, default=0.9, help="SGD momentum.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.option("--training_data", type=click.STRING, default='C:/Users/aawang/CovidTraffic/Data/saltlake_week.csv',
              help="path to training data")
def run(training_data, epochs, batch_size, learning_rate, momentum, seed):
    warnings.filterwarnings("ignore")
    saltlake_week = pd.read_csv(training_data)

    X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                             'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                             'Stay at Home', 'Mask', 'School Opening', 'Health Emergency', 'Holiday']].values[:156, :]

    data = X_total[54:, :]
    scaler, values = utils.scale(data)
    values = utils.series_to_supervised(values, n_in=config.N_WEEKS, n_out=1, dropnan=True).values

    y_scaler, y = utils.scale(data[:, 1].reshape((len(data), 1)))

    train = values[:88, :]
    valid = values[88:92]
    test = values[92:, :]
    print(train.shape)
    print(valid.shape)
    print(test.shape)

    y = values[:, -config.N_FEATURES:]

    X_train = train[:, :config.N_OBS]
    y_train = train[:, -config.N_FEATURES:][:, 1]
    X_valid = valid[:, :config.N_OBS]
    y_valid = valid[:, -config.N_FEATURES:][:, 1]
    X_test = test[:, :config.N_OBS]
    y_test = test[:, -config.N_FEATURES:][:, 1]

    X_train = X_train.reshape((X_train.shape[0], config.N_WEEKS, config.N_FEATURES))
    X_valid = X_valid.reshape((X_valid.shape[0], config.N_WEEKS, config.N_FEATURES))
    X_test = X_test.reshape((X_test.shape[0], config.N_WEEKS, config.N_FEATURES))

    with mlflow.start_run():
        mlflow.keras.autolog()
        if epochs == 0:  # score null model
            eval_and_log_metrics(
                "train", y_train, np.ones(len(y_train)) * np.mean(y_train), epoch=-1
            )
            eval_and_log_metrics("val", y_valid, np.ones(len(y_valid)) * np.mean(y_valid), epoch=-1)
            eval_and_log_metrics("test", y_test, np.ones(len(y_test)) * np.mean(y_test), epoch=-1)
        else:
            with MLflowCheckpoint(X_test, y_test) as mlflow_logger:
                es = EarlyStopping(
                    monitor="val_loss",
                    patience=config.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True)

                model = keras.Sequential()
                model.add(keras.layers.LSTM(30, input_shape=config.INPUT_SHAPE, activation='relu'))
                model.add(keras.layers.Dropout(0.2))
                model.add(keras.layers.Dense(1))
                model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01),
                              metrics=['mse'])
                model.build()

                print("[INFO] training the model...")
                history = model.fit(
                    x=X_train, y=y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=config.BS,
                    verbose=1,
                    callbacks=[es, mlflow_logger],
                    epochs=config.EPOCHS)

                print("[INFO] evaluating network...")
                predictions = model.predict(x=X_test)
                print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

                utils.save_plot(history, "output/loss_plot.png")

                y_test_inv = utils.invert_scale(y_scaler, y_test.reshape((len(y_test), 1)))
                predictions_inv = y_scaler.inverse_transform(predictions)

                eval_and_log_metrics("test", y_test_inv, predictions_inv, epoch=2)

                X = values[:, :config.N_OBS]
                y = values[:, -config.N_FEATURES:][:, 1]

                m, n = X.shape

                X = X.reshape((X.shape[0], config.N_WEEKS, config.N_FEATURES))
                total_pred = model.predict(x=X)

                mse_test = mean_squared_error(y_test_inv, predictions_inv)
                rmse_test = mean_squared_error(y_test_inv, predictions_inv, squared=False)
                mape_test = mean_absolute_percentage_error(y_test_inv, predictions_inv)

                print("Mean Squared Error: {}".format(mse_test))
                print("Root Mean Squared Error: {}".format(rmse_test))
                print("Mean Absolute Percentage Error: {}".format(mape_test))


if __name__ == "__main__":
    run()

    # mlflow run -e train_search_mlflow_test.py --experiment-id 1 Code/LSTM --no-conda
