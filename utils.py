import pandas as pd

from sklearn import preprocessing
from tensorflow import keras

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')


def save_plot(H, path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="Train Loss")
    plt.plot(H.history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss (Mean Squared Error)")
    plt.legend()
    plt.savefig(path)


def plot_predicted(true, predicted, path):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(true, label='True')
    plt.plot(predicted, label='Predicted')
    plt.xlabel("Week")
    plt.ylabel("VMT")
    plt.legend()
    plt.savefig(path)


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


def graph_data(df, groups):
    plt.figure()
    for group in groups:
        plt.plot(df[group])
    plt.show()


def scale(X_total):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_total)
    X_total_scaled = scaler.transform(X_total)
    return scaler, X_total_scaled


def invert_scale(scaler, data):
    inv_data = scaler.inverse_transform(data)
    return inv_data
