import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import Model, initializers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn import preprocessing

import stellargraph as sg
from stellargraph.layer import GCN_LSTM

mpl.use('TkAgg')


def sequence_data_preparation(seq_len, pre_len, data):
    X, y = [], []

    for i in range(data.shape[1] - int(seq_len + pre_len - 1)):
        a = data[:, i: i + seq_len + pre_len]
        X.append(a[:, :seq_len])
        y.append(a[:, -1])

    X = np.array(X)
    y = np.array(y)

    return X, y


def save_plot(H, path):
    # plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], "b-", label="Train Loss")
    plt.plot(H.history["val_loss"], "r-", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Mean Squared Error)")
    plt.legend(loc='best')
    plt.savefig(path)


def plot_predicted(true, predicted, type, path, test=True):
    # plt.style.use('ggplot')
    plt.figure()
    plt.plot(true, "b-", label='True')
    plt.plot(predicted, "r-", label=type)
    plt.title("Predicted VMT for Salt Lake County")
    if test:
        plt.xticks(ticks=np.arange(4), labels=np.arange(1, 5, dtype=int))
    plt.xlabel("Week")
    plt.ylabel("VMT (Tens of Millions Miles)")
    plt.legend(loc='best')
    plt.savefig(path)


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to model")
ap.add_argument("-p", "--plot", required=True, help="path to output loss plot")
args = vars(ap.parse_args())

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

es = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True)

saltlake_week = pd.read_csv('Data/saltlake_week.csv')

saltlake_week.fillna(0, inplace=True)

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                         'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                         'Stay at Home', 'Mask', 'School Opening', 'Health Emergency']].values

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_total[59:, :])
X_total_scaled = scaler.transform(X_total[59:, :])
data = X_total_scaled.T
print(data.shape)
X, y = sequence_data_preparation(1, 1, data)

X_train = X[:88, :, :]
y_train = y[:88,:]
X_test = X[88:, :, :]
y_test = y[88:, :]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

rmse_benchmark = math.sqrt(mean_squared_error(X_total[148:, 1] / 1000000, X_total[147:151, 1] / 1000000))
mape_benchmark = math.sqrt(mean_absolute_percentage_error(X_total[148:, 1] / 1000000, X_total[147:151, 1] / 1000000))

rmse_ref = rmse_benchmark
mape_ref = mape_benchmark

print('RMSE Benchmark: {}'.format(rmse_ref))
print('MAPE Benchmark: {}'.format(mape_ref))

# Load saved model
model = keras.models.load_model(args['model'])

print("[INFO] model summary")
print(model.summary())
model.reset_states()

# Evaluate model
print("[INFO] evaluating network...")
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))
model.reset_states()
# save_plot(history, args["plot"])

# Make predictions
ythat = model.predict(X_train, batch_size=5)
model.reset_states()
yhat = model.predict(X_test, batch_size=5)
model.reset_states()

# Rescale values
train_rescref = scaler.inverse_transform(ythat)
test_rescref = scaler.inverse_transform(yhat)

# Evaluate results
test_pred = test_rescref[:, 1]
test_true = X_total[148:, 1]

train_pred = train_rescref[:, 1]
train_true = X_total[60:148, 1]

trainScore = math.sqrt(mean_squared_error(test_true / 1000000, test_pred / 1000000))
trainScore_mape = mean_absolute_percentage_error(test_true / 1000000, test_pred / 1000000)

print("Root Mean Squared Error: {}".format(trainScore))
print("Mean Absolute Percentage Error: {}".format(trainScore_mape))

# Visualize predictions
fig1 = plt.figure(figsize=(15, 8))
ax1 = fig1.add_subplot(1, 1, 1)
plt.plot(test_pred, "r-", label="Predicted")
plt.plot(test_true, "b-", label="True")
plt.xlabel("Week")
plt.ylabel("VMT (Veh-Miles)")
plt.legend(loc="best", fontsize=10)
plt.show()

fig2 = plt.figure(figsize=(15, 8))
ax2 = fig2.add_subplot(1, 1, 1)
plt.plot(train_pred, "r-", label="Predicted")
plt.plot(train_true, "b-", label="True")
plt.xlabel("Week")
plt.ylabel("VMT (Veh-Miles)")
plt.legend(loc="best", fontsize=10)
plt.show()

plot_predicted(test_true, test_pred, "GCN-LSTM", "figures/gcn-lstm_test.png")
plot_predicted(train_true, train_pred, "GCN-LSTM", "figures/gcn-lstm_train.png", test=False)

# python evaluate_GCN-LSTM.py --model GCN-LSTM/Model/GCN-LSTM-20 --plot figures/gcn-lstm_plot.png
