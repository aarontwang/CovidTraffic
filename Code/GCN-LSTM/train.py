import argparse
import math
import os
import random
import config
import utils

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from keras import Model, initializers
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import stellargraph as sg
from stellargraph.layer import GCN_LSTM

mpl.use('TkAgg')

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", required=True, help="path to output loss plot")
args = vars(ap.parse_args())

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

saltlake_week = pd.read_csv('../Data/saltlake_week.csv')

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                         'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                         'Stay at Home', 'Mask', 'School Opening', 'Health Emergency']].values[:156, :]

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_total[58:, :])
X_total_scaled = scaler.transform(X_total[58:, :])
data = X_total_scaled.T

X, y = utils.sequence_data_preparation(config.SEQ_LEN, config.PRE_LEN, data)

X_train = X[:88, :, :]
y_train = y[:88, :]
X_test = X[88:, :, :]
y_test = y[88:, :]

es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

model = GCN_LSTM(
    seq_len=config.SEQ_LEN,
    adj=config.TOTAL_ADJACENT_MATRIX,
    gc_layer_sizes=[16],
    gc_activations=['relu'],
    lstm_layer_sizes=[60],
    lstm_activations=['relu'],
    kernel_initializer=initializers.Identity(gain=1)
)

x_input, x_output = model.in_out_tensors()
model = Model(inputs=x_input, outputs=x_output)
lr = 0.001
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse",
              metrics=["mse"])
print(model.summary())

print("[INFO] training the best model...")
H = model.fit(
    x=X_train, y=y_train,
    validation_split=0.2,
    batch_size=config.BS,
    callbacks=[es],
    epochs=config.EPOCHS
)

print("[INFO] evaluating network...")
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

utils.save_plot(H, args["plot"])
model.save('Model/GCN-LSTM')

ythat = model.predict(X_train)
yhat = model.predict(X_test)

# Rescale values
train_rescref = scaler.inverse_transform(ythat)
test_rescref = scaler.inverse_transform(yhat)

test_pred = test_rescref[:, 1]
test_true = X_total[147:, 1]

train_pred = train_rescref[:, 1]
train_true = X_total[60:147, 1]

mse_test = mean_squared_error(test_true, test_pred)
rmse_test = mean_squared_error(test_true, test_pred, squared=False)
mape_test = mean_absolute_percentage_error(test_true, test_pred)

print("Root Mean Squared Error: {}".format(rmse_test))
print("Mean Absolute Percentage Error: {}".format(mape_test))

# all test result visualization
fig1 = plt.figure(figsize=(15, 8))
ax1 = fig1.add_subplot(1, 1, 1)
plt.plot(test_pred, "r-", label="prediction")
plt.plot(test_true, "b-", label="true")
plt.xlabel("time")
plt.ylabel("vmt")
plt.legend(loc="best", fontsize=10)
plt.show()

ax2 = fig1.add_subplot(2, 1, 1)
plt.plot(train_pred, "r-", label="prediction")
plt.plot(train_true, "b-", label="true")
plt.xlabel("time")
plt.ylabel("vmt")
plt.legend(loc="best", fontsize=10)
plt.show()

utils.plot_predicted(test_true, test_pred, "output/vmt_test_predicted.png")
utils.plot_predicted(train_true, train_pred, "output/vmt_train_predicted.png")

# python train.py --plot output/loss_plot.png
