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
    plt.figure(figsize=(16, 9))
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
model = keras.models.load_model(args['model'])

print("[INFO] model summary")
print(model.summary())

# Evaluate model
print("[INFO] evaluating network...")
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

# save_plot(history, args["plot"])

# Make predictions
ythat = model.predict(X_train, batch_size=4)
model.reset_states()
yhat = model.predict(X_test, batch_size=4)
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
plt.plot(test_pred, "r-", label="LSTM")
plt.plot(test_true, "b-", label="True")
plt.xticks(ticks=np.arange(4), labels=np.arange(1, 5, dtype=int))
plt.xlabel("Week")
plt.ylabel("VMT (Tens of Millions Miles)")
plt.legend(loc="best", fontsize=10)
plt.show()

fig2 = plt.figure(figsize=(15, 8))
ax2 = fig2.add_subplot(1, 1, 1)
plt.plot(train_pred, "r-", label="LSTM")
plt.plot(train_true, "b-", label="True")
plt.xlabel("Week")
plt.ylabel("VMT (Tens of Millions Miles)")
plt.legend(loc="best", fontsize=10)
plt.show()

plot_predicted(test_true, test_pred, 'LSTM', "figures/lstm_test.png")
plot_predicted(train_true, train_pred, 'LSTM', "figures/lstm_train.png", test=False)

# python evaluate_LSTM.py --model LSTM/Model/LSTM-15 --plot figures/lstm_plot.png
