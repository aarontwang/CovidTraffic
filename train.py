import argparse
import config
import math
import os
import random
import utils

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping

from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tuner", required=True, type=str, choices=['bayesian'], help="type of hyperparameter tuner")
ap.add_argument("-p", "--plot", required=True, help="path to output loss plot")
args = vars(ap.parse_args())

seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

saltlake_week = pd.read_csv('Data/saltlake_week (3).csv')

saltlake_week.fillna(0, inplace=True)

utils.graph_data(saltlake_week, ['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate',
                                 'Percent_Fully_Vaccinated_5&Older'])

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP',
                         'SNOW', 'SNWD', 'Percent_Fully_Vaccinated_5&Older', 'TAVG', 'Stay at Home',
                         'Mask', 'School Opening', 'Health Emergency']]

scaler, data = utils.scale(X_total)
data = utils.series_to_supervised(data, n_in=config.N_WEEKS, n_out=1, dropnan=True)
values = data.values
X = values[:, :config.N_OBS]
y = values[:, -config.N_FEATURES:]

m, n = X.shape

train = values[:147, :]
test = values[147:, :]

X_train = train[:, :config.N_OBS]
y_train = train[:, -config.N_FEATURES:]

X_test = test[:, :config.N_OBS]
y_test = test[:, -config.N_FEATURES:]

X_train = X_train.reshape((X_train.shape[0], config.N_WEEKS, config.N_FEATURES))
X_test = X_test.reshape((X_test.shape[0], config.N_WEEKS, config.N_FEATURES))

rmse_benchmark = math.sqrt(mean_squared_error(X_total.iloc[148:, 1] / 1000000, X_total.iloc[146:150, 1] / 1000000))
mape_benchmark = math.sqrt(
    mean_absolute_percentage_error(X_total.iloc[148:, 1] / 1000000, X_total.iloc[146:150, 1] / 1000000))

rmse_ref = rmse_benchmark
mape_ref = mape_benchmark

print('RMSE Benchmark: {}'.format(rmse_ref))
print('MAPE Benchmark: {}'.format(mape_ref))

es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

if args["tuner"] == "bayesian":
    print("[INFO] instantiating a bayesian optimization tuner object...")
    tuner = kt.BayesianOptimization(
        build_model,
        objective="mse",
        max_trials=10,
        seed=42,
        directory=config.OUTPUT_PATH,
        project_name=args["tuner"])
else:
    print("[INFO] instantiating a hyperband tuner object...")
    tuner = kt.Hyperband(
        build_model,
        objective="mse",
        max_trials=10,
        seed=42,
        directory=config.OUTPUT_PATH,
        project_name=args["tuner"])

print("[INFO] performing hyperparameter search...")
tuner.search(
    x=X_train, y=y_train,
    # validation_data=(X_test, y_test),
    validation_split=0.2,
    batch_size=config.BS,
    callbacks=[es],
    epochs=config.EPOCHS
)

bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("[INFO] optimal number of nodes in lstm_1 layer: {}".format(
    bestHP.get("lstm_1")))
print("[INFO] optimal learning rate: {}".format(
    bestHP.get("learning_rate")))

print("[INFO] training the best model...")
model = tuner.hypermodel.build(bestHP)
H = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_test, y_test),
    batch_size=config.BS,
    callbacks=[es],
    epochs=config.EPOCHS
)

print("[INFO] evaluating network...")
predictions = model.predict(x=X_test, batch_size=config.BS)
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

utils.save_plot(H, args["plot"])

y_test_inv = utils.invert_scale(scaler, y_test)[:, 1]
predictions_inv = utils.invert_scale(scaler, predictions)[:, 1]

X = X.reshape((X.shape[0], config.N_WEEKS, config.N_FEATURES))
total_pred = model.predict(x=X, batch_size=config.BS)

y_inv = utils.invert_scale(scaler, y)[:, 1]
total_pred_inv = utils.invert_scale(scaler, total_pred)[:, 1]

rmse_test = mean_squared_error(y_test_inv, predictions_inv, squared=False)
mape_test = mean_absolute_percentage_error(y_test_inv, predictions_inv)

print("Root Mean Squared Error: {}".format(rmse_test))
print("Mean Absolute Percentage Error: {}".format(mape_test))

utils.plot_predicted(y_test_inv, predictions_inv, "output/vmt_test_predicted.png")
utils.plot_predicted(y_inv, total_pred_inv, "output/vmt_total_predicted.png")

# python train.py --tuner bayesian --plot output/bayesian_plot.png
