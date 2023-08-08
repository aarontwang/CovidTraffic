import numpy as np
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def plot_predicted(true, predicted, title, path):
    plt.figure()
    plt.title(title)
    plt.plot(true, label='True')
    plt.plot(predicted, label='Predicted')
    plt.xlabel("Week")
    plt.ylabel("VMT")
    plt.legend()
    plt.savefig(path)


def sequence_data_preparation(seq_len, pre_len, data):
    X, y = [], []

    for i in range(data.shape[1] - int(seq_len + pre_len - 1)):
        a = data[:, i: i + seq_len + pre_len]
        X.append(a[:, :seq_len])
        y.append(a[:, -1])

    X = np.array(X)
    y = np.array(y)

    return X, y


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
