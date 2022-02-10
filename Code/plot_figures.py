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


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


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


def plot_predicted(true, lstm, gcn, persistence, path, test=True):
    # plt.style.use('ggplot')
    plt.figure(figsize=(12, 9))
    mpl.rc('font', family='Times New Roman', size=14)
    plt.plot(true, "b-", label='True')
    plt.plot(lstm, "r-", label='LSTM')
    plt.plot(gcn, "g-", label='GCN-LSTM')
    plt.plot(persistence, color="tab:orange", label="Persistence")
    mpl.rc('font', family='Times New Roman', size=18)
    plt.title("Predicted VMT for Salt Lake County")
    mpl.rc('font', family='Times New Roman', size=14)
    if test:
        plt.xticks(ticks=np.arange(4), labels=np.arange(1, 5, dtype=int))
    plt.xlabel("Week")
    plt.ylabel("VMT (Tens of Millions Miles)")
    plt.legend(loc='best')
    plt.savefig(path)


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, help="path to model")
ap.add_argument("-p", "--plot", required=False, help="path to output loss plot")
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

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNOW',
                         'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                         'Stay at Home', 'Mask', 'School Opening', 'Health Emergency']].values

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_total[59:, :])
X_total_scaled = scaler.transform(X_total[59:, :])
data = series_to_supervised(X_total_scaled, n_in=1, n_out=1).values

print(data.shape)

X = data[:, :12]
y = data[:, -12:]

X_train = X[:88, :]
y_train = y[:88, :]
X_test = X[88:, :]
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

X_train = X_train.reshape((X_train.shape[0], 1, 12))
X_test = X_test.reshape((X_test.shape[0], 1, 12))

# Load saved model
lstm = keras.models.load_model('LSTM/Model/LSTM-15')

# Make predictions
lstm_ythat = lstm.predict(X_train, batch_size=4)
lstm.reset_states()
lstm_yhat = lstm.predict(X_test, batch_size=4)
lstm.reset_states()

# Rescale values
lstm_train_rescref = scaler.inverse_transform(lstm_ythat)
lstm_test_rescref = scaler.inverse_transform(lstm_yhat)

test_true = X_total[148:, 1]
train_true = X_total[60:148, 1]

# Evaluate results
lstm_test_pred = lstm_test_rescref[:, 1]
lstm_train_pred = lstm_train_rescref[:, 1]

lstm_trainScore = math.sqrt(mean_squared_error(test_true / 1000000, lstm_test_pred / 1000000))
lstm_trainScore_mape = mean_absolute_percentage_error(test_true / 1000000, lstm_test_pred / 1000000)

print("LSTM Root Mean Squared Error: {}".format(lstm_trainScore))
print("LSTM Mean Absolute Percentage Error: {}".format(lstm_trainScore_mape))

# GCN-LSTM
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

# Load saved model
gcn = keras.models.load_model('GCN-LSTM/Model/GCN-LSTM-20')

# Make predictions
gcn_ythat = gcn.predict(X_train, batch_size=5)
gcn.reset_states()
gcn_yhat = gcn.predict(X_test, batch_size=5)
gcn.reset_states()

# Rescale values
gcn_train_rescref = scaler.inverse_transform(gcn_ythat)
gcn_test_rescref = scaler.inverse_transform(gcn_yhat)

# Evaluate results
gcn_test_pred = gcn_test_rescref[:, 1]
gcn_train_pred = gcn_train_rescref[:, 1]

gcn_trainScore = math.sqrt(mean_squared_error(test_true / 1000000, gcn_test_pred / 1000000))
gcn_trainScore_mape = mean_absolute_percentage_error(test_true / 1000000, gcn_test_pred / 1000000)

print("GCN-LSTM Root Mean Squared Error: {}".format(gcn_trainScore))
print("GCN-LSTM Mean Absolute Percentage Error: {}".format(gcn_trainScore_mape))


plot_predicted(test_true, lstm_test_pred, gcn_test_pred, X_total[147:151, 1], "figures/eval_test.png")
plot_predicted(train_true, lstm_train_pred, gcn_train_pred, X_total[59:147, 1], "figures/eval_train.png", test=False)

# python plot_figures.py
